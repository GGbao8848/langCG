import { Bot, Loader2, PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen, Send } from "lucide-react";
import React, { useEffect, useRef, useState } from "react";
import { ChatMessage } from "./components/ChatMessage";
import { ChatSidebar } from "./components/ChatSidebar";
import { ToolSidebar } from "./components/ToolSidebar";
import { fetchModels, fetchTools, sendChat } from "./services/agentApi";
import { ChatModelOption, ChatSession, UIMessage } from "./types";

const newSession = (): ChatSession => ({
  id: crypto.randomUUID(),
  name: "新对话",
  updatedAt: Date.now(),
  messages: [],
});

const selectionKey = (provider: string, model: string) => `${provider}::${model}`;

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([newSession()]);
  const [currentSessionId, setCurrentSessionId] = useState<string>(() => "");
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isChatSidebarOpen, setIsChatSidebarOpen] = useState(true);
  const [isToolSidebarOpen, setIsToolSidebarOpen] = useState(false);
  const [toolSearchQuery, setToolSearchQuery] = useState("");
  const [tools, setTools] = useState<any[]>([]);
  const [modelOptions, setModelOptions] = useState<ChatModelOption[]>([]);
  const [activeProvider, setActiveProvider] = useState("openrouter");
  const [activeModel, setActiveModel] = useState("openrouter/free");

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const currentSession = sessions.find((session) => session.id === currentSessionId) ?? sessions[0];
  const activeModelLabel =
    modelOptions.find((option) => option.provider === activeProvider && option.model === activeModel)?.label ??
    `${activeProvider}: ${activeModel}`;

  const filteredTools = tools.filter((tool) => {
    const query = toolSearchQuery.toLowerCase();
    return tool.name.toLowerCase().includes(query) || tool.description.toLowerCase().includes(query);
  });

  useEffect(() => {
    setCurrentSessionId((current) => current || sessions[0].id);
  }, [sessions]);

  useEffect(() => {
    fetchModels()
      .then((data) => {
        setModelOptions(data.models);
        const saved = localStorage.getItem("llmSelection");
        const defaultSelection = saved ? JSON.parse(saved) : data.default;
        setActiveProvider(defaultSelection.provider);
        setActiveModel(defaultSelection.model);
      })
      .catch((error) => {
        console.error(error);
      });

    fetchTools()
      .then((data) => setTools(data.tools))
      .catch((error) => {
        console.error(error);
      });
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentSession.messages]);

  const updateSession = (id: string, fields: Partial<ChatSession>) => {
    setSessions((previous) =>
      previous.map((session) => (session.id === id ? { ...session, ...fields, updatedAt: Date.now() } : session)),
    );
  };

  const handleModelChange = (value: string) => {
    const [provider, model] = value.split("::");
    setActiveProvider(provider);
    setActiveModel(model);
    localStorage.setItem("llmSelection", JSON.stringify({ provider, model }));
  };

  const handleSend = async (event?: React.FormEvent) => {
    event?.preventDefault();
    if (!input.trim() || isLoading) return;

    const sessionId = currentSession.id;
    const userText = input.trim();
    setInput("");
    setIsLoading(true);

    const userMessage: UIMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: userText,
    };
    const nextMessages = [...currentSession.messages, userMessage];
    updateSession(sessionId, {
      messages: nextMessages,
      name:
        currentSession.messages.length === 0 && currentSession.name === "新对话"
          ? `${userText.slice(0, 15)}${userText.length > 15 ? "..." : ""}`
          : currentSession.name,
    });

    try {
      const response = await sendChat({
        provider: activeProvider,
        model: activeModel,
        messages: nextMessages,
      });
      const modelMessage: UIMessage = {
        id: crypto.randomUUID(),
        role: "model",
        text: response.text || "模型没有返回文本。",
        toolCalls: response.toolCalls,
      };
      updateSession(sessionId, { messages: [...nextMessages, modelMessage] });
    } catch (error: any) {
      updateSession(sessionId, {
        messages: [
          ...nextMessages,
          {
            id: crypto.randomUUID(),
            role: "model",
            text: `Error: ${error.message ?? "请求失败"}`,
          },
        ],
      });
    } finally {
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

            <select
              aria-label="选择 LLM 模型"
              value={selectionKey(activeProvider, activeModel)}
              onChange={(event) => handleModelChange(event.target.value)}
              className="min-w-0 max-w-[52vw] cursor-pointer appearance-none truncate rounded-xl border border-slate-200 bg-white/90 px-4 py-2 text-sm font-medium text-slate-700 shadow-sm backdrop-blur-md transition-colors hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 sm:max-w-none"
            >
              {modelOptions.map((option) => (
                <option key={selectionKey(option.provider, option.model)} value={selectionKey(option.provider, option.model)}>
                  {option.label}
                </option>
              ))}
            </select>
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

          <div className="scrollbar-thin flex flex-1 flex-col overflow-y-auto px-4 pb-40 pt-20 sm:px-16 md:px-28 lg:px-44">
            <div className={currentSession.messages.length === 0 ? "flex flex-1 items-center justify-center" : "space-y-6"}>
              {currentSession.messages.length === 0 ? (
                <div className="max-w-xl text-center text-slate-400">
                  <Bot className="mx-auto mb-4 h-10 w-10 text-slate-300" />
                  <h1 className="text-lg font-semibold text-slate-700">langCG Agent</h1>
                  <p className="mt-2 text-sm">选择模型后开始对话，工具会由后端 agent 自动调用。</p>
                </div>
              ) : (
                currentSession.messages.map((message) => <ChatMessage key={message.id} msg={message} />)
              )}

              {isLoading && (
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

          <div className="pointer-events-none absolute bottom-0 left-0 right-0 z-10 bg-gradient-to-t from-slate-50/95 via-slate-50/80 to-transparent px-4 pb-6 pt-12 backdrop-blur-[2px] sm:px-16 md:px-28 lg:px-44">
            <form
              onSubmit={handleSend}
              className="pointer-events-auto relative flex items-end overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-lg transition-all hover:border-slate-300 focus-within:border-indigo-400 focus-within:ring-2 focus-within:ring-indigo-500/50"
            >
              <textarea
                className="max-h-48 min-h-14 flex-1 resize-none bg-transparent py-4 pl-6 pr-14 leading-relaxed text-slate-800 outline-none placeholder:text-slate-400"
                placeholder="输入消息..."
                value={input}
                rows={1}
                onChange={(event) => {
                  setInput(event.target.value);
                  event.target.style.height = "auto";
                  event.target.style.height = `${event.target.scrollHeight}px`;
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    handleSend(event);
                  }
                }}
                disabled={isLoading}
                autoFocus
              />
              <button
                aria-label="发送消息"
                type="submit"
                disabled={!input.trim() || isLoading}
                className="absolute bottom-3 right-3 rounded-full bg-indigo-600 p-2 text-white transition-colors hover:bg-indigo-700 disabled:bg-slate-200 disabled:text-slate-400"
              >
                {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="ml-0.5 h-4 w-4" />}
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
