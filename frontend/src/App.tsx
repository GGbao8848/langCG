import { Bot, Loader2, PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen, Send, Square } from "lucide-react";
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { ChatMessage } from "./components/ChatMessage";
import { ChatSidebar } from "./components/ChatSidebar";
import { ToolSidebar } from "./components/ToolSidebar";
import {
  fetchLLMSettings,
  fetchTools,
  fetchUserSettings,
  saveLLMSettings,
  saveUserSettings,
  streamChat,
  testLLMSettings,
  testUserSettings,
  testYoloEnvironment,
} from "./services/agentApi";
import { loadPersistedChatState, savePersistedChatState, type PersistedChatState } from "./services/chatStorage";
import { AgentStreamEvent, ChatSession, LLMProvider, LLMSettings, ToolCallData, UIMessage, UserSettings } from "./types";

const newSession = (): ChatSession => ({
  id: crypto.randomUUID(),
  name: "新对话",
  updatedAt: Date.now(),
  messages: [],
});

const defaultUserSettings: UserSettings = {
  remote_sftp_host: "172.31.1.42",
  remote_sftp_username: "",
  remote_sftp_private_key_path: "/home/qzq/.ssh/id_ed25519",
  remote_sftp_port: 22,
  local_yolo_train_venv_path: "",
};

const defaultLLMSettings: LLMSettings = {
  active_provider: "ollama",
  providers: {
    ollama: {
      model: "gemma4:e4b",
      api_key: "",
      base_url: "http://127.0.0.1:11434",
    },
    openrouter: {
      model: "openrouter/free",
      api_key: "",
      base_url: "https://openrouter.ai/api/v1",
    },
  },
  model_options: {
    ollama: ["gemma4:e4b", "qwen3.5:9b"],
    openrouter: ["openrouter/free"],
  },
};

function normalizeSessionsForStorage(sessions: ChatSession[]): ChatSession[] {
  const normalized = sessions
    .filter((session): session is ChatSession => {
      return (
        typeof session?.id === "string" &&
        typeof session?.name === "string" &&
        typeof session?.updatedAt === "number" &&
        Array.isArray(session?.messages)
      );
    })
    .map((session) => ({
      ...session,
      messages: session.messages
        .filter((message) => message?.role === "user" || message?.role === "model")
        .map((message) => ({
          ...message,
          toolCalls: message.toolCalls?.map((toolCall) =>
            toolCall.status === "running"
              ? { ...toolCall, status: "canceled" as const, result: toolCall.result ?? "刷新时已中止" }
              : toolCall,
          ),
        })),
    }));

  return normalized.length > 0 ? normalized : [newSession()];
}

function remoteUserSettingsKey(settings: UserSettings) {
  return JSON.stringify({
    remote_sftp_host: settings.remote_sftp_host.trim(),
    remote_sftp_username: settings.remote_sftp_username.trim(),
    remote_sftp_private_key_path: settings.remote_sftp_private_key_path.trim(),
    remote_sftp_port: Number(settings.remote_sftp_port || 0),
  });
}

function isCompleteUserSettings(settings: UserSettings) {
  return Boolean(
    settings.remote_sftp_host.trim() &&
      settings.remote_sftp_username.trim() &&
      settings.remote_sftp_private_key_path.trim() &&
      Number(settings.remote_sftp_port) > 0,
  );
}

function yoloEnvironmentKey(settings: UserSettings) {
  return settings.local_yolo_train_venv_path.trim();
}

function llmSettingsKey(settings: LLMSettings) {
  const activeSettings = activeLLMProviderSettings(settings);
  return JSON.stringify({
    active_provider: settings.active_provider,
    model: activeSettings.model.trim(),
    api_key: settings.active_provider === "openrouter" ? activeSettings.api_key.trim() : "",
    base_url: activeSettings.base_url.trim(),
  });
}

function isCompleteLLMSettings(settings: LLMSettings) {
  const activeSettings = activeLLMProviderSettings(settings);
  return Boolean(
    settings.active_provider &&
      activeSettings.model.trim() &&
      activeSettings.base_url.trim() &&
      (settings.active_provider !== "openrouter" || activeSettings.api_key.trim()),
  );
}

function providerLabel(provider: LLMProvider) {
  return provider === "openrouter" ? "OpenRouter" : "Ollama";
}

function activeLLMProviderSettings(settings: LLMSettings) {
  return settings.providers[settings.active_provider] ?? defaultLLMSettings.providers[settings.active_provider];
}

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>(() => [newSession()]);
  const [currentSessionId, setCurrentSessionId] = useState<string>(() => "");
  const [isPersistenceReady, setIsPersistenceReady] = useState(false);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isChatSidebarOpen, setIsChatSidebarOpen] = useState(true);
  const [isToolSidebarOpen, setIsToolSidebarOpen] = useState(false);
  const [toolSearchQuery, setToolSearchQuery] = useState("");
  const [tools, setTools] = useState<any[]>([]);
  const [llmSettings, setLLMSettings] = useState<LLMSettings>(defaultLLMSettings);
  const [savedLLMSettings, setSavedLLMSettings] = useState<LLMSettings>(defaultLLMSettings);
  const [userSettings, setUserSettings] = useState<UserSettings>(defaultUserSettings);
  const [isSavingLLMSettings, setIsSavingLLMSettings] = useState(false);
  const [isSavingRemoteUserSettings, setIsSavingRemoteUserSettings] = useState(false);
  const [isSavingYoloEnvironment, setIsSavingYoloEnvironment] = useState(false);
  const [isTestingLLMSettings, setIsTestingLLMSettings] = useState(false);
  const [isTestingUserSettings, setIsTestingUserSettings] = useState(false);
  const [isTestingYoloEnvironment, setIsTestingYoloEnvironment] = useState(false);
  const [llmSettingsStatus, setLLMSettingsStatus] = useState("");
  const [userSettingsStatus, setUserSettingsStatus] = useState("");
  const [yoloEnvironmentStatus, setYoloEnvironmentStatus] = useState("");
  const [testedLLMSettingsKey, setTestedLLMSettingsKey] = useState("");
  const [testedUserSettingsKey, setTestedUserSettingsKey] = useState("");
  const [testedYoloEnvironmentKey, setTestedYoloEnvironmentKey] = useState("");
  const [savedLLMSettingsKey, setSavedLLMSettingsKey] = useState("");
  const [savedRemoteUserSettingsKey, setSavedRemoteUserSettingsKey] = useState("");
  const [savedYoloEnvironmentKey, setSavedYoloEnvironmentKey] = useState("");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageInputRef = useRef<HTMLTextAreaElement>(null);
  const activeAbortControllerRef = useRef<AbortController | null>(null);
  const latestPersistenceStateRef = useRef<PersistedChatState | null>(null);

  const currentSession = sessions.find((session) => session.id === currentSessionId) ?? sessions[0];
  const lastMessage = currentSession.messages[currentSession.messages.length - 1];
  const lastMessageHasVisibleContent = Boolean(lastMessage?.text || lastMessage?.toolCalls?.length);
  const savedActiveLLMProviderSettings = activeLLMProviderSettings(savedLLMSettings);
  const activeModelLabel = `${providerLabel(savedLLMSettings.active_provider)}: ${savedActiveLLMProviderSettings.model || "未配置"}`;

  const filteredTools = tools.filter((tool) => {
    const query = toolSearchQuery.toLowerCase();
    return tool.name.toLowerCase().includes(query) || tool.description.toLowerCase().includes(query);
  });
  const isLLMSettingsComplete = isCompleteLLMSettings(llmSettings);
  const isLLMSettingsTestPassed = testedLLMSettingsKey === llmSettingsKey(llmSettings);
  const isLLMSettingsSaved = savedLLMSettingsKey === llmSettingsKey(llmSettings);
  const isUserSettingsComplete = isCompleteUserSettings(userSettings);
  const isUserSettingsTestPassed = testedUserSettingsKey === remoteUserSettingsKey(userSettings);
  const isYoloEnvironmentTestPassed =
    Boolean(yoloEnvironmentKey(userSettings)) && testedYoloEnvironmentKey === yoloEnvironmentKey(userSettings);
  const isUserSettingsSaved = savedRemoteUserSettingsKey === remoteUserSettingsKey(userSettings);
  const isYoloEnvironmentSaved = savedYoloEnvironmentKey === yoloEnvironmentKey(userSettings);

  useEffect(() => {
    setCurrentSessionId((current) => {
      if (current && sessions.some((session) => session.id === current)) {
        return current;
      }
      return sessions[0].id;
    });
  }, [sessions]);

  useEffect(() => {
    let isMounted = true;

    loadPersistedChatState()
      .then((state) => {
        if (!isMounted) return;
        if (!state) {
          setIsPersistenceReady(true);
          return;
        }

        const normalizedSessions = normalizeSessionsForStorage(state.sessions);
        setSessions(normalizedSessions);
        setCurrentSessionId(
          normalizedSessions.some((session) => session.id === state.currentSessionId)
            ? state.currentSessionId
            : normalizedSessions[0].id,
        );
        setIsPersistenceReady(true);
      })
      .catch((error) => {
        console.warn("Failed to initialize chat persistence", error);
        if (isMounted) setIsPersistenceReady(true);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!isPersistenceReady || !currentSessionId) return;

    const persistedState = {
      sessions: normalizeSessionsForStorage(sessions),
      currentSessionId,
      savedAt: Date.now(),
    };
    latestPersistenceStateRef.current = persistedState;

    const saveTimer = window.setTimeout(() => {
      savePersistedChatState(persistedState);
    }, 250);

    return () => window.clearTimeout(saveTimer);
  }, [currentSessionId, isPersistenceReady, sessions]);

  useEffect(() => {
    const flushLatestState = () => {
      const state = latestPersistenceStateRef.current;
      if (state) {
        savePersistedChatState({ ...state, savedAt: Date.now() });
      }
    };

    window.addEventListener("pagehide", flushLatestState);
    return () => window.removeEventListener("pagehide", flushLatestState);
  }, []);

  useEffect(() => {
    let isMounted = true;

    fetchLLMSettings()
      .then((settings) => {
        if (!isMounted) return;
        setLLMSettings(settings);
        setSavedLLMSettings(settings);
        setSavedLLMSettingsKey(llmSettingsKey(settings));
        if (isCompleteLLMSettings(settings)) {
          setIsTestingLLMSettings(true);
          setLLMSettingsStatus("正在自动测试当前模型联通...");
          testLLMSettings(settings)
            .then((result) => {
              if (!isMounted) return;
              setTestedLLMSettingsKey(llmSettingsKey(settings));
              setLLMSettingsStatus(result.message || "当前模型联通正常");
            })
            .catch((error: any) => {
              if (!isMounted) return;
              setTestedLLMSettingsKey("");
              setLLMSettingsStatus(error?.message ?? "当前模型联通测试失败");
            })
            .finally(() => {
              if (isMounted) setIsTestingLLMSettings(false);
            });
        }
      })
      .catch((error) => {
        console.error(error);
      });

    fetchTools()
      .then((data) => {
        if (!isMounted) return;
        setTools(data.tools);
      })
      .catch((error) => {
        console.error(error);
      });

    fetchUserSettings()
      .then((settings) => {
        if (!isMounted) return;
        setUserSettings(settings);
        setSavedRemoteUserSettingsKey(remoteUserSettingsKey(settings));
        setSavedYoloEnvironmentKey(yoloEnvironmentKey(settings));

        if (isCompleteUserSettings(settings)) {
          setIsTestingUserSettings(true);
          setUserSettingsStatus("正在自动测试 SSH/SFTP 联通...");
          testUserSettings(settings)
            .then((result) => {
              if (!isMounted) return;
              setTestedUserSettingsKey(remoteUserSettingsKey(settings));
              setUserSettingsStatus(result.message || "联通正常");
            })
            .catch((error: any) => {
              if (!isMounted) return;
              setTestedUserSettingsKey("");
              setUserSettingsStatus(error?.message ?? "联通测试失败");
            })
            .finally(() => {
              if (isMounted) setIsTestingUserSettings(false);
            });
        }

        if (yoloEnvironmentKey(settings)) {
          setIsTestingYoloEnvironment(true);
          setYoloEnvironmentStatus("正在自动测试Ultralytics YOLO CLI...");
          testYoloEnvironment(settings)
            .then((result) => {
              if (!isMounted) return;
              setTestedYoloEnvironmentKey(yoloEnvironmentKey(settings));
              setYoloEnvironmentStatus(result.message || "Ultralytics YOLO CLI正常");
            })
            .catch((error: any) => {
              if (!isMounted) return;
              setTestedYoloEnvironmentKey("");
              setYoloEnvironmentStatus(error?.message ?? "YOLO CLI测试失败");
            })
            .finally(() => {
              if (isMounted) setIsTestingYoloEnvironment(false);
            });
        }
      })
      .catch((error) => {
        console.error(error);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentSession.messages]);

  useLayoutEffect(() => {
    const textarea = messageInputRef.current;
    if (!textarea) return;

    textarea.style.height = "auto";
    if (input) {
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [input]);

  const updateSession = (id: string, fields: Partial<ChatSession>) => {
    setSessions((previous) =>
      previous.map((session) => (session.id === id ? { ...session, ...fields, updatedAt: Date.now() } : session)),
    );
  };

  const updateMessage = (sessionId: string, messageId: string, updater: (message: UIMessage) => UIMessage) => {
    setSessions((previous) =>
      previous.map((session) => {
        if (session.id !== sessionId) return session;
        return {
          ...session,
          updatedAt: Date.now(),
          messages: session.messages.map((message) => (message.id === messageId ? updater(message) : message)),
        };
      }),
    );
  };

  const upsertToolCall = (
    message: UIMessage,
    toolCall: ToolCallData,
    mode: "append-or-replace" | "result",
  ): UIMessage => {
    const existingToolCalls = message.toolCalls ?? [];
    const index = existingToolCalls.findIndex((item) => item.id === toolCall.id);
    if (index === -1) {
      return { ...message, toolCalls: [...existingToolCalls, toolCall] };
    }

    return {
      ...message,
      toolCalls: existingToolCalls.map((item, itemIndex) =>
        itemIndex === index
          ? {
              ...item,
              ...toolCall,
              name: toolCall.name || item.name,
              args: mode === "result" ? item.args : toolCall.args,
            }
          : item,
      ),
    };
  };

  const applyStreamEvent = (sessionId: string, messageId: string, event: AgentStreamEvent) => {
    if (event.type === "metadata") return;

    updateMessage(sessionId, messageId, (message) => {
      if (event.type === "token") {
        return { ...message, text: `${message.isProgress ? "" : message.text ?? ""}${event.text}`, isProgress: false };
      }

      if (event.type === "progress") {
        return { ...message, text: event.message, isProgress: true };
      }

      if (event.type === "tool_call") {
        return upsertToolCall(
          message,
          {
            id: event.id,
            name: event.name,
            args: event.args,
            status: "running",
          },
          "append-or-replace",
        );
      }

      if (event.type === "tool_result") {
        return upsertToolCall(
          message,
          {
            id: event.id,
            name: event.name ?? "tool",
            status: event.status,
            result: event.result,
          },
          "result",
        );
      }

      if (event.type === "error") {
        return { ...message, text: `${message.isProgress ? "" : message.text ?? ""}\nError: ${event.message}`.trim(), isProgress: false };
      }

      if (event.type === "done" && event.text && (!message.text || message.isProgress)) {
        return { ...message, text: event.text, isProgress: false };
      }

      return message;
    });
  };

  const markMessageCanceled = (sessionId: string, messageId: string) => {
    updateMessage(sessionId, messageId, (message) => {
      const toolCalls = message.toolCalls?.map((toolCall) =>
        toolCall.status === "running"
          ? { ...toolCall, status: "canceled" as const, result: toolCall.result ?? "已终止" }
          : toolCall,
      );
      const suffix = "已终止当前任务。";
      return {
        ...message,
        text: message.text ? `${message.text}\n\n${suffix}` : suffix,
        isProgress: false,
        toolCalls,
      };
    });
  };

  const stopCurrentTask = () => {
    activeAbortControllerRef.current?.abort();
  };

  const handleLLMSettingsChange = (nextSettings: LLMSettings) => {
    const currentKey = llmSettingsKey(llmSettings);
    const nextKey = llmSettingsKey(nextSettings);
    setLLMSettings(nextSettings);
    if (currentKey !== nextKey) {
      setTestedLLMSettingsKey("");
      setLLMSettingsStatus("");
    }
  };

  const handleUserSettingsChange = (nextSettings: UserSettings) => {
    const currentRemoteUserSettingsKey = remoteUserSettingsKey(userSettings);
    const nextRemoteUserSettingsKey = remoteUserSettingsKey(nextSettings);
    const currentYoloEnvironmentKey = yoloEnvironmentKey(userSettings);
    const nextYoloEnvironmentKey = yoloEnvironmentKey(nextSettings);
    setUserSettings(nextSettings);
    if (currentRemoteUserSettingsKey !== nextRemoteUserSettingsKey) {
      setTestedUserSettingsKey("");
      setSavedRemoteUserSettingsKey("");
      setUserSettingsStatus("");
    }
    if (currentYoloEnvironmentKey !== nextYoloEnvironmentKey) {
      setTestedYoloEnvironmentKey("");
      setSavedYoloEnvironmentKey("");
      setYoloEnvironmentStatus("");
    }
  };

  const handleTestUserSettings = async () => {
    if (!isCompleteUserSettings(userSettings)) {
      setTestedUserSettingsKey("");
      setUserSettingsStatus("请先填写完整 Host、Username、Private key、Port");
      return;
    }

    setIsTestingUserSettings(true);
    setTestedUserSettingsKey("");
    setUserSettingsStatus("正在测试 SSH/SFTP 联通...");
    try {
      const result = await testUserSettings(userSettings);
      setTestedUserSettingsKey(remoteUserSettingsKey(userSettings));
      setUserSettingsStatus(result.message || "联通正常");
    } catch (error: any) {
      setUserSettingsStatus(error?.message ?? "联通测试失败");
    } finally {
      setIsTestingUserSettings(false);
    }
  };

  const handleTestLLMSettings = async () => {
    if (!isCompleteLLMSettings(llmSettings)) {
      setTestedLLMSettingsKey("");
      setLLMSettingsStatus(
        llmSettings.active_provider === "openrouter"
          ? "请先填写 Provider、Model、Base URL、API Key"
          : "请先填写 Provider、Model、Ollama URL",
      );
      return;
    }

    setIsTestingLLMSettings(true);
    setTestedLLMSettingsKey("");
    setLLMSettingsStatus("正在测试当前模型联通...");
    try {
      const result = await testLLMSettings(llmSettings);
      setTestedLLMSettingsKey(llmSettingsKey(llmSettings));
      setLLMSettingsStatus(result.message || "当前模型联通正常");
    } catch (error: any) {
      setLLMSettingsStatus(error?.message ?? "当前模型联通测试失败");
    } finally {
      setIsTestingLLMSettings(false);
    }
  };

  const handleTestYoloEnvironment = async () => {
    const envKey = yoloEnvironmentKey(userSettings);
    if (!envKey) {
      setTestedYoloEnvironmentKey("");
      setYoloEnvironmentStatus("请先填写YOLO训练虚拟环境路径");
      return;
    }

    setIsTestingYoloEnvironment(true);
    setTestedYoloEnvironmentKey("");
    setYoloEnvironmentStatus("正在测试Ultralytics YOLO CLI...");
    try {
      const result = await testYoloEnvironment(userSettings);
      setTestedYoloEnvironmentKey(envKey);
      setYoloEnvironmentStatus(result.message || "Ultralytics YOLO CLI正常");
    } catch (error: any) {
      setYoloEnvironmentStatus(error?.message ?? "YOLO CLI测试失败");
    } finally {
      setIsTestingYoloEnvironment(false);
    }
  };

  const handleSaveRemoteUserSettings = async () => {
    setIsSavingRemoteUserSettings(true);
    try {
      const savedSettings = await saveUserSettings(userSettings);
      setUserSettings(savedSettings);
      setSavedRemoteUserSettingsKey(remoteUserSettingsKey(savedSettings));
      setUserSettingsStatus("已保存，后续远端发布会使用该配置");
    } catch (error: any) {
      setUserSettingsStatus(error?.message ?? "保存失败");
    } finally {
      setIsSavingRemoteUserSettings(false);
    }
  };

  const handleSaveLLMSettings = async () => {
    if (!isLLMSettingsTestPassed) {
      setLLMSettingsStatus("请先测试当前配置，联通正常后再保存");
      return;
    }

    setIsSavingLLMSettings(true);
    try {
      const savedSettings = await saveLLMSettings(llmSettings);
      setLLMSettings(savedSettings);
      setSavedLLMSettings(savedSettings);
      setSavedLLMSettingsKey(llmSettingsKey(savedSettings));
      setLLMSettingsStatus("已保存，后续对话会使用该模型配置");
    } catch (error: any) {
      setLLMSettingsStatus(error?.message ?? "保存失败");
    } finally {
      setIsSavingLLMSettings(false);
    }
  };

  const handleSaveYoloEnvironment = async () => {
    setIsSavingYoloEnvironment(true);
    try {
      const savedSettings = await saveUserSettings(userSettings);
      setUserSettings(savedSettings);
      setSavedYoloEnvironmentKey(yoloEnvironmentKey(savedSettings));
      setYoloEnvironmentStatus("已保存，后续本地训练会使用该配置");
    } catch (error: any) {
      setYoloEnvironmentStatus(error?.message ?? "保存失败");
    } finally {
      setIsSavingYoloEnvironment(false);
    }
  };

  const handleSend = async (event?: React.FormEvent) => {
    event?.preventDefault();
    if (isLoading) {
      stopCurrentTask();
      return;
    }
    if (!input.trim() || isLoading || !savedActiveLLMProviderSettings.model.trim()) return;

    const sessionId = currentSession.id;
    const userText = input.trim();
    const abortController = new AbortController();
    activeAbortControllerRef.current = abortController;
    setInput("");
    setIsLoading(true);

    const userMessage: UIMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: userText,
    };
    const nextMessages = [...currentSession.messages, userMessage];
    const assistantMessageId = crypto.randomUUID();
    const assistantMessage: UIMessage = {
      id: assistantMessageId,
      role: "model",
      text: "",
      toolCalls: [],
    };
    updateSession(sessionId, {
      messages: [...nextMessages, assistantMessage],
      name:
        currentSession.messages.length === 0 && currentSession.name === "新对话"
          ? `${userText.slice(0, 15)}${userText.length > 15 ? "..." : ""}`
          : currentSession.name,
    });

    try {
      await streamChat(
        {
          provider: savedLLMSettings.active_provider,
          model: savedActiveLLMProviderSettings.model.trim(),
          messages: nextMessages,
          signal: abortController.signal,
        },
        (streamEvent) => applyStreamEvent(sessionId, assistantMessageId, streamEvent),
      );
    } catch (error: any) {
      if (error.name === "AbortError") {
        markMessageCanceled(sessionId, assistantMessageId);
        return;
      }

      updateMessage(sessionId, assistantMessageId, (message) => {
        return {
          ...message,
          text: `${message.text ?? ""}\nError: ${error.message ?? "请求失败"}`.trim(),
        };
      });
    } finally {
      if (activeAbortControllerRef.current === abortController) {
        activeAbortControllerRef.current = null;
      }
      setIsLoading(false);
    }
  };

  const createNewSession = () => {
    const emptySession = sessions.find((session) => session.messages.length === 0);
    if (emptySession) {
      setCurrentSessionId(emptySession.id);
      return;
    }

    const session = newSession();
    setSessions((previous) => [...previous, session]);
    setCurrentSessionId(session.id);
  };

  const deleteSession = (id: string) => {
    setSessions((previous) => {
      const remaining = previous.filter((session) => session.id !== id);
      if (remaining.length === 0) {
        const session = newSession();
        setCurrentSessionId(session.id);
        return [session];
      }
      if (currentSessionId === id) {
        setCurrentSessionId(remaining[0].id);
      }
      return remaining;
    });
  };

  const renameSession = (id: string, name: string) => {
    updateSession(id, { name });
  };

  return (
    <div className="flex h-dvh flex-col bg-slate-50 font-sans text-slate-900">
      <div className="relative flex min-h-0 flex-1 overflow-hidden">
        <ChatSidebar
          isOpen={isChatSidebarOpen}
          sessions={sessions}
          currentSessionId={currentSession.id}
          onSelectSession={setCurrentSessionId}
          onNewSession={createNewSession}
          onDeleteSession={deleteSession}
          onRenameSession={renameSession}
          activeModelLabel={activeModelLabel}
          llmSettings={llmSettings}
          isLLMSettingsComplete={isLLMSettingsComplete}
          isSavingLLMSettings={isSavingLLMSettings}
          isTestingLLMSettings={isTestingLLMSettings}
          isLLMSettingsTestPassed={isLLMSettingsTestPassed}
          isLLMSettingsSaved={isLLMSettingsSaved}
          llmSettingsStatus={llmSettingsStatus}
          onLLMSettingsChange={handleLLMSettingsChange}
          onTestLLMSettings={handleTestLLMSettings}
          onSaveLLMSettings={handleSaveLLMSettings}
          userSettings={userSettings}
          isUserSettingsComplete={isUserSettingsComplete}
          isSavingRemoteUserSettings={isSavingRemoteUserSettings}
          isSavingYoloEnvironment={isSavingYoloEnvironment}
          isTestingUserSettings={isTestingUserSettings}
          isTestingYoloEnvironment={isTestingYoloEnvironment}
          isUserSettingsTestPassed={isUserSettingsTestPassed}
          isUserSettingsSaved={isUserSettingsSaved}
          isYoloEnvironmentTestPassed={isYoloEnvironmentTestPassed}
          isYoloEnvironmentSaved={isYoloEnvironmentSaved}
          userSettingsStatus={userSettingsStatus}
          yoloEnvironmentStatus={yoloEnvironmentStatus}
          onUserSettingsChange={handleUserSettingsChange}
          onTestUserSettings={handleTestUserSettings}
          onTestYoloEnvironment={handleTestYoloEnvironment}
          onSaveUserSettings={handleSaveRemoteUserSettings}
          onSaveYoloEnvironment={handleSaveYoloEnvironment}
        />

        <main className="relative flex min-w-0 flex-1 flex-col">
          <div className="absolute left-4 top-4 z-40 flex max-w-[calc(100%-2rem)] items-center gap-2">
            <button
              aria-label="Toggle chat sidebar"
              onClick={() => setIsChatSidebarOpen((open) => !open)}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-200 bg-white/90 text-slate-500 shadow-sm backdrop-blur-md transition-colors hover:bg-slate-100 hover:text-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
            >
              {isChatSidebarOpen ? <PanelLeftClose className="h-5 w-5" /> : <PanelLeftOpen className="h-5 w-5" />}
            </button>
          </div>

          <div className="absolute right-4 top-4 z-40 rounded-xl border border-slate-200 bg-white/90 p-1 shadow-sm backdrop-blur-md">
            <button
              aria-label="Toggle tools sidebar"
              onClick={() => setIsToolSidebarOpen((open) => !open)}
              className="flex h-9 w-9 items-center justify-center rounded-lg text-slate-500 transition-colors hover:bg-slate-100 hover:text-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
            >
              {isToolSidebarOpen ? <PanelRightClose className="h-5 w-5" /> : <PanelRightOpen className="h-5 w-5" />}
            </button>
          </div>

          <div className="scrollbar-thin flex flex-1 flex-col overflow-y-auto px-4 pb-40 pt-20">
            <div className={`mx-auto w-full max-w-4xl ${currentSession.messages.length === 0 ? "flex flex-1 items-center justify-center" : "space-y-6"}`}>
              {currentSession.messages.length === 0 ? (
                <div className="max-w-xl text-center text-slate-400">
                  <Bot className="mx-auto mb-4 h-10 w-10 text-slate-300" />
                  <h1 className="text-lg font-semibold text-slate-700">langCG Agent</h1>
                  <p className="mt-2 text-sm">选择模型后开始对话，工具会由后端 agent 自动调用。</p>
                </div>
              ) : (
                currentSession.messages
                  .filter((message) => message.role === "user" || message.text || message.toolCalls?.length)
                  .map((message) => <ChatMessage key={message.id} msg={message} />)
              )}

              {isLoading && !(lastMessage?.role === "model" && lastMessageHasVisibleContent) && (
                <div className="flex items-start">
                  <div className="mr-3 mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-500 shadow-sm">
                    <Bot className="h-5 w-5 text-white" />
                  </div>
                  <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 shadow-sm">
                    <Loader2 className="h-4 w-4 animate-spin text-slate-400" />
                    <span className="text-sm text-slate-500">{activeModelLabel}</span>
                  </div>
                </div>
              )}
            </div>
            <div ref={messagesEndRef} />
          </div>

          <div className="pointer-events-none absolute bottom-0 left-0 right-0 z-10 bg-slate-50 px-4 pb-6 pt-4">
            <form
              onSubmit={handleSend}
              className="pointer-events-auto relative mx-auto flex max-w-4xl items-end overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-lg transition-all hover:border-slate-300 focus-within:border-indigo-400 focus-within:ring-2 focus-within:ring-indigo-500/50"
            >
              <textarea
                ref={messageInputRef}
                className="scrollbar-thin max-h-48 min-h-14 flex-1 resize-none overflow-y-auto bg-transparent py-4 pl-6 pr-14 leading-relaxed text-slate-800 outline-none placeholder:text-slate-400"
                placeholder="输入消息..."
                value={input}
                rows={1}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    handleSend(event);
                  }
                }}
                autoFocus
              />
              <button
                aria-label={isLoading ? "终止当前任务" : "发送消息"}
                type="submit"
                disabled={!isLoading && (!input.trim() || !savedActiveLLMProviderSettings.model.trim())}
                className={`absolute bottom-3 right-3 rounded-full p-2 text-white transition-colors disabled:bg-slate-200 disabled:text-slate-400 ${
                  isLoading ? "bg-rose-600 hover:bg-rose-700" : "bg-indigo-600 hover:bg-indigo-700"
                }`}
              >
                {isLoading ? <Square className="h-4 w-4 fill-current" /> : <Send className="ml-0.5 h-4 w-4" />}
              </button>
            </form>
          </div>
        </main>

        <ToolSidebar
          isOpen={isToolSidebarOpen}
          toolSearchQuery={toolSearchQuery}
          setToolSearchQuery={setToolSearchQuery}
          filteredTools={filteredTools}
        />
      </div>
    </div>
  );
}
