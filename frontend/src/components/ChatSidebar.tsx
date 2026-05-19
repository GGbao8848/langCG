import React, { useEffect, useState } from "react";
import { MessageSquare, Plus, MoreHorizontal, Pencil, Trash2, X, Check, ChevronDown, ChevronRight } from "lucide-react";
import { ChatSession, LLMProvider, LLMSettings, UserSettings } from "../types";

interface ChatSidebarProps {
  isOpen: boolean;
  sessions: ChatSession[];
  currentSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, newName: string) => void;
  activeModelLabel?: string;
  llmSettings: LLMSettings;
  isLLMSettingsComplete: boolean;
  isSavingLLMSettings: boolean;
  isTestingLLMSettings: boolean;
  isLLMSettingsTestPassed: boolean;
  isLLMSettingsSaved: boolean;
  llmSettingsStatus: string;
  onLLMSettingsChange: (settings: LLMSettings) => void;
  onTestLLMSettings: () => void;
  onSaveLLMSettings: () => void;
  userSettings: UserSettings;
  isUserSettingsComplete: boolean;
  isSavingRemoteUserSettings: boolean;
  isSavingYoloEnvironment: boolean;
  isTestingUserSettings: boolean;
  isTestingYoloEnvironment: boolean;
  isUserSettingsTestPassed: boolean;
  isUserSettingsSaved: boolean;
  isYoloEnvironmentTestPassed: boolean;
  isYoloEnvironmentSaved: boolean;
  userSettingsStatus: string;
  yoloEnvironmentStatus: string;
  onUserSettingsChange: (settings: UserSettings) => void;
  onTestUserSettings: () => void;
  onTestYoloEnvironment: () => void;
  onSaveUserSettings: () => void;
  onSaveYoloEnvironment: () => void;
}

export const ChatSidebar: React.FC<ChatSidebarProps> = ({
  isOpen,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onRenameSession,
  activeModelLabel,
  llmSettings,
  isLLMSettingsComplete,
  isSavingLLMSettings,
  isTestingLLMSettings,
  isLLMSettingsTestPassed,
  isLLMSettingsSaved,
  llmSettingsStatus,
  onLLMSettingsChange,
  onTestLLMSettings,
  onSaveLLMSettings,
  userSettings,
  isUserSettingsComplete,
  isSavingRemoteUserSettings,
  isSavingYoloEnvironment,
  isTestingUserSettings,
  isTestingYoloEnvironment,
  isUserSettingsTestPassed,
  isUserSettingsSaved,
  isYoloEnvironmentTestPassed,
  isYoloEnvironmentSaved,
  userSettingsStatus,
  yoloEnvironmentStatus,
  onUserSettingsChange,
  onTestUserSettings,
  onTestYoloEnvironment,
  onSaveUserSettings,
  onSaveYoloEnvironment,
}) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [menuPos, setMenuPos] = useState<{ id: string, top: number, left: number } | null>(null);
  const [isLLMSettingsCollapsed, setIsLLMSettingsCollapsed] = useState(true);
  const [isUserSettingsCollapsed, setIsUserSettingsCollapsed] = useState(true);
  const [isYoloEnvironmentCollapsed, setIsYoloEnvironmentCollapsed] = useState(true);

  const startEditing = (id: string, currentName: string) => {
    setEditingId(id);
    setEditName(currentName);
    setMenuPos(null);
  };

  const handleRename = () => {
    if (editingId && editName.trim()) {
      onRenameSession(editingId, editName.trim());
    }
    setEditingId(null);
  };

  const sortedSessions = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);
  const llmSettingsCardClass = isLLMSettingsComplete
    ? isLLMSettingsTestPassed
      ? "border-emerald-200 bg-emerald-50/60"
      : "border-red-300 bg-red-50/80"
    : "border-red-300 bg-red-50/80";
  const userSettingsCardClass = isUserSettingsComplete
    ? isUserSettingsTestPassed
      ? "border-emerald-200 bg-emerald-50/60"
      : "border-red-300 bg-red-50/80"
    : "border-red-300 bg-red-50/80";
  const hasYoloEnvironment = Boolean(userSettings.local_yolo_train_venv_path.trim());
  const yoloEnvironmentCardClass =
    hasYoloEnvironment && isYoloEnvironmentTestPassed
      ? "border-emerald-200 bg-emerald-50/60"
      : "border-red-300 bg-red-50/80";
  const userSettingsInputClass = (isFilled: boolean) =>
    `mt-1 w-full rounded-lg border px-2 py-1.5 text-xs text-slate-800 outline-none focus:ring-1 ${
      isFilled
        ? "border-slate-200 bg-white focus:border-indigo-400 focus:ring-indigo-400"
        : "border-red-300 bg-red-50 focus:border-red-400 focus:ring-red-400"
    }`;
  const userSettingsSubtitle = isUserSettingsComplete
    ? isUserSettingsTestPassed
      ? isUserSettingsSaved
        ? "已测试并保存"
        : "已测试，待保存"
      : "待测试"
    : "信息未填写完整";
  const yoloEnvironmentSubtitle = hasYoloEnvironment
    ? isYoloEnvironmentTestPassed
      ? isYoloEnvironmentSaved
        ? "已测试并保存"
        : "已测试，待保存"
      : "待测试"
    : "未配置";
  const llmSettingsSubtitle = isLLMSettingsComplete
    ? isLLMSettingsTestPassed
      ? isLLMSettingsSaved
        ? "已测试并保存"
        : "已测试，待保存"
      : "待测试"
    : "模型配置未填写完整";
  const activeLLMProviderSettings = llmSettings.providers[llmSettings.active_provider];
  const activeLLMModelOptions = llmSettings.model_options[llmSettings.active_provider] ?? [];
  const llmProviderLabel = llmSettings.active_provider === "openrouter" ? "OpenRouter" : "Ollama";
  const llmBaseUrlLabel = llmSettings.active_provider === "openrouter" ? "OpenRouter Base URL" : "Ollama URL";
  const updateActiveLLMProviderSettings = (updates: Partial<typeof activeLLMProviderSettings>) => {
    onLLMSettingsChange({
      ...llmSettings,
      providers: {
        ...llmSettings.providers,
        [llmSettings.active_provider]: {
          ...activeLLMProviderSettings,
          ...updates,
        },
      },
    });
  };

  useEffect(() => {
    if (isLLMSettingsSaved && isLLMSettingsTestPassed) {
      setIsLLMSettingsCollapsed(true);
    }
  }, [isLLMSettingsSaved, isLLMSettingsTestPassed]);

  useEffect(() => {
    if (isUserSettingsSaved && isUserSettingsTestPassed) {
      setIsUserSettingsCollapsed(true);
    }
  }, [isUserSettingsSaved, isUserSettingsTestPassed]);

  useEffect(() => {
    if (isYoloEnvironmentSaved && isYoloEnvironmentTestPassed) {
      setIsYoloEnvironmentCollapsed(true);
    }
  }, [isYoloEnvironmentSaved, isYoloEnvironmentTestPassed]);

  return (
    <div style={{ width: isOpen ? '280px' : '0' }} className="bg-slate-50 border-r border-slate-200 transition-all duration-300 flex flex-col shrink-0 overflow-hidden relative z-30">
      <div className="w-[280px] flex-1 flex flex-col min-h-0 absolute inset-0">
        <div className="pt-5 pb-2 px-4 shrink-0 flex justify-center">
          <button
            onClick={onNewSession}
            className="w-[90%] flex items-center justify-center gap-2 py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl transition-colors text-sm font-medium shadow-sm shadow-indigo-200/50"
          >
            <Plus className="w-4 h-4 text-white" />
            新会话
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto scrollbar-thin px-3 pb-3 space-y-1">
          {sortedSessions.map((session) => (
            <div
              key={session.id}
              className={`group relative flex items-center p-2 rounded-lg cursor-pointer transition-colors ${
                currentSessionId === session.id
                  ? "bg-indigo-50 text-indigo-700"
                  : "hover:bg-slate-200 text-slate-700"
              }`}
              onClick={() => {
                if (editingId !== session.id) {
                  onSelectSession(session.id);
                }
              }}
            >
              <MessageSquare className={`w-4 h-4 mr-3 shrink-0 ${currentSessionId === session.id ? 'text-indigo-500' : 'text-slate-400'}`} />
              
              {editingId === session.id ? (
                <div className="flex-1 flex items-center min-w-0 mr-1 gap-1">
                  <input
                    type="text"
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleRename();
                      if (e.key === 'Escape') setEditingId(null);
                    }}
                    autoFocus
                    className="flex-1 min-w-0 bg-white border border-indigo-300 rounded px-1.5 py-0.5 text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500 text-slate-800"
                    onClick={(e) => e.stopPropagation()}
                  />
                  <button onClick={(e) => { e.stopPropagation(); handleRename(); }} className="text-emerald-600 hover:bg-emerald-100 p-1 rounded shrink-0">
                    <Check className="w-3 h-3" />
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); setEditingId(null); }} className="text-slate-500 hover:bg-slate-200 p-1 rounded shrink-0">
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ) : (
                <div className="flex-1 truncate text-sm select-none" title={session.name}>
                  {session.name}
                </div>
              )}

              {editingId !== session.id && (
                <div className="relative shrink-0 ml-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (menuPos?.id === session.id) {
                        setMenuPos(null);
                      } else {
                        const rect = e.currentTarget.getBoundingClientRect();
                        setMenuPos({
                          id: session.id,
                          top: rect.bottom + 4,
                          left: rect.right - 128, // w-32 is 128px
                        });
                      }
                    }}
                    className={`p-1 rounded-md transition-opacity ${menuPos?.id === session.id ? 'opacity-100 bg-slate-300' : 'opacity-0 group-hover:opacity-100 hover:bg-slate-300'}`}
                  >
                    <MoreHorizontal className="w-4 h-4 text-slate-500" />
                  </button>

                  {menuPos?.id === session.id && (
                    <>
                      <div className="fixed inset-0 z-40" onClick={(e) => { e.stopPropagation(); setMenuPos(null); }}></div>
                      <div 
                        className="fixed w-32 py-1 bg-white border border-slate-200 rounded-lg shadow-lg z-50 text-sm"
                        style={{ top: menuPos.top, left: menuPos.left }}
                      >
                        <button
                          className="w-full text-left px-3 py-1.5 text-slate-700 hover:bg-slate-50 flex items-center"
                          onClick={(e) => {
                            e.stopPropagation();
                            startEditing(session.id, session.name);
                          }}
                        >
                          <Pencil className="w-3.5 h-3.5 mr-2" /> 命名
                        </button>
                        <button
                          className="w-full text-left px-3 py-1.5 text-red-600 hover:bg-red-50 flex items-center"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteSession(session.id);
                            setMenuPos(null);
                          }}
                        >
                          <Trash2 className="w-3.5 h-3.5 mr-2" /> 删除
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
        
        <div className="p-3 border-t border-slate-200 shrink-0">
          <div className={`rounded-2xl border p-3 shadow-sm transition-colors ${llmSettingsCardClass}`}>
            <div className={`flex items-center justify-between ${isLLMSettingsCollapsed ? "" : "mb-2"}`}>
              <button
                type="button"
                onClick={() => setIsLLMSettingsCollapsed((collapsed) => !collapsed)}
                className="flex min-w-0 flex-1 items-center text-left focus:outline-none"
              >
                <span className="mr-1.5 shrink-0 text-slate-500">
                  {isLLMSettingsCollapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                </span>
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-slate-800">当前模型</div>
                  <div className={`truncate text-[10px] ${isLLMSettingsTestPassed ? "text-emerald-700" : "text-red-600"}`}>
                    {isLLMSettingsCollapsed ? activeModelLabel ?? "OpenRouter / Ollama" : llmSettingsSubtitle}
                  </div>
                </div>
              </button>
              {!isLLMSettingsCollapsed && (
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={onTestLLMSettings}
                    disabled={!isLLMSettingsComplete || isTestingLLMSettings || isSavingLLMSettings}
                    className="rounded-lg bg-indigo-600 px-2.5 py-1 text-xs font-medium text-white transition-colors hover:bg-indigo-700 disabled:bg-slate-300 disabled:text-slate-500"
                  >
                    {isTestingLLMSettings ? "测试中" : "测试"}
                  </button>
                  <button
                    onClick={onSaveLLMSettings}
                    disabled={!isLLMSettingsTestPassed || isSavingLLMSettings || isTestingLLMSettings}
                    className="rounded-lg bg-slate-900 px-2.5 py-1 text-xs font-medium text-white transition-colors hover:bg-slate-700 disabled:bg-slate-300 disabled:text-slate-500"
                  >
                    {isSavingLLMSettings ? "保存中" : "保存"}
                  </button>
                </div>
              )}
            </div>

            {!isLLMSettingsCollapsed && (
              <>
                <div className={`text-[10px] ${isLLMSettingsComplete ? "text-slate-500" : "text-red-500"}`}>
                  {llmSettingsSubtitle}
                </div>
                <div className="mt-2 space-y-2">
                  <label className="block text-[11px] font-medium text-slate-500">
                    Provider
                    <select
                      value={llmSettings.active_provider}
                      onChange={(event) => {
                        const provider = event.target.value as LLMProvider;
                        onLLMSettingsChange({
                          ...llmSettings,
                          active_provider: provider,
                        });
                      }}
                      className={userSettingsInputClass(true)}
                    >
                      <option value="ollama">Ollama</option>
                      <option value="openrouter">OpenRouter</option>
                    </select>
                  </label>
                  <label className="block text-[11px] font-medium text-slate-500">
                    Model
                    <select
                      value={activeLLMProviderSettings.model}
                      onChange={(event) => updateActiveLLMProviderSettings({ model: event.target.value })}
                      className={userSettingsInputClass(Boolean(activeLLMProviderSettings.model.trim()))}
                    >
                      {activeLLMModelOptions.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="block text-[11px] font-medium text-slate-500">
                    {llmBaseUrlLabel}
                    <input
                      value={activeLLMProviderSettings.base_url}
                      placeholder={llmSettings.active_provider === "openrouter" ? "https://openrouter.ai/api/v1" : "http://127.0.0.1:11434"}
                      onChange={(event) => updateActiveLLMProviderSettings({ base_url: event.target.value })}
                      className={`${userSettingsInputClass(Boolean(activeLLMProviderSettings.base_url.trim()))} placeholder:text-slate-300`}
                    />
                  </label>
                  {llmSettings.active_provider === "openrouter" && (
                    <label className="block text-[11px] font-medium text-slate-500">
                      API Key
                      <input
                        type="password"
                        value={activeLLMProviderSettings.api_key}
                        placeholder="sk-or-..."
                        onChange={(event) => updateActiveLLMProviderSettings({ api_key: event.target.value })}
                        className={`${userSettingsInputClass(Boolean(activeLLMProviderSettings.api_key.trim()))} placeholder:text-slate-300`}
                      />
                    </label>
                  )}
                </div>
                <div className={`mt-2 text-[10px] ${isLLMSettingsTestPassed ? "text-emerald-700" : "text-red-600"}`}>
                  {llmSettingsStatus ||
                    (isLLMSettingsComplete ? "请先点击测试，联通正常后才能保存" : `${llmProviderLabel} 配置未填写完整`)}
                </div>
              </>
            )}
          </div>

          <div className={`mt-3 rounded-2xl border p-3 shadow-sm transition-colors ${userSettingsCardClass}`}>
            <div className={`flex items-center justify-between ${isUserSettingsCollapsed ? "" : "mb-2"}`}>
              <button
                type="button"
                onClick={() => setIsUserSettingsCollapsed((collapsed) => !collapsed)}
                className="flex min-w-0 flex-1 items-center text-left focus:outline-none"
              >
                <span className="mr-1.5 shrink-0 text-slate-500">
                  {isUserSettingsCollapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                </span>
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-slate-800">用户信息</div>
                  {isUserSettingsCollapsed && (
                    <div className={`truncate text-[10px] ${isUserSettingsTestPassed ? "text-emerald-700" : "text-red-600"}`}>
                      {isUserSettingsComplete
                        ? isUserSettingsTestPassed
                          ? `${userSettings.remote_sftp_username}@${userSettings.remote_sftp_host}:${userSettings.remote_sftp_port}`
                          : userSettingsSubtitle
                        : "信息未填写完整"}
                    </div>
                  )}
                </div>
              </button>
              {!isUserSettingsCollapsed && (
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={onTestUserSettings}
                    disabled={!isUserSettingsComplete || isTestingUserSettings || isSavingRemoteUserSettings}
                    className="rounded-lg bg-indigo-600 px-2.5 py-1 text-xs font-medium text-white transition-colors hover:bg-indigo-700 disabled:bg-slate-300 disabled:text-slate-500"
                  >
                    {isTestingUserSettings ? "测试中" : "测试"}
                  </button>
                  <button
                    onClick={onSaveUserSettings}
                    disabled={isSavingRemoteUserSettings || isTestingUserSettings}
                    className="rounded-lg bg-slate-900 px-2.5 py-1 text-xs font-medium text-white transition-colors hover:bg-slate-700 disabled:bg-slate-300 disabled:text-slate-500"
                  >
                    {isSavingRemoteUserSettings ? "保存中" : "保存"}
                  </button>
                </div>
              )}
            </div>

            {!isUserSettingsCollapsed && (
              <>
                <div className={`text-[10px] ${isUserSettingsComplete ? "text-slate-500" : "text-red-500"}`}>
                  {userSettingsSubtitle}
                </div>

            <div className="mt-2 space-y-2">
              <label className="block text-[11px] font-medium text-slate-500">
                Host
                <input
                  value={userSettings.remote_sftp_host}
                  onChange={(event) =>
                    onUserSettingsChange({ ...userSettings, remote_sftp_host: event.target.value })
                  }
                  className={userSettingsInputClass(Boolean(userSettings.remote_sftp_host.trim()))}
                />
              </label>
              <label className="block text-[11px] font-medium text-slate-500">
                Username
                <input
                  value={userSettings.remote_sftp_username}
                  placeholder="留空，使用时再填写"
                  onChange={(event) =>
                    onUserSettingsChange({ ...userSettings, remote_sftp_username: event.target.value })
                  }
                  className={`${userSettingsInputClass(Boolean(userSettings.remote_sftp_username.trim()))} placeholder:text-slate-300`}
                />
              </label>
              <label className="block text-[11px] font-medium text-slate-500">
                Private key
                <input
                  value={userSettings.remote_sftp_private_key_path}
                  onChange={(event) =>
                    onUserSettingsChange({ ...userSettings, remote_sftp_private_key_path: event.target.value })
                  }
                  className={userSettingsInputClass(Boolean(userSettings.remote_sftp_private_key_path.trim()))}
                />
              </label>
              <label className="block text-[11px] font-medium text-slate-500">
                Port
                <input
                  type="number"
                  min={1}
                  max={65535}
                  value={userSettings.remote_sftp_port}
                  onChange={(event) =>
                    onUserSettingsChange({
                      ...userSettings,
                      remote_sftp_port: Number(event.target.value || 22),
                    })
                  }
                  className={userSettingsInputClass(Number(userSettings.remote_sftp_port) > 0)}
                />
              </label>
            </div>
            <div
              className={`mt-2 text-[10px] ${
                isUserSettingsTestPassed ? "text-emerald-700" : "text-red-600"
              }`}
            >
              {userSettingsStatus ||
                (isUserSettingsComplete ? "请先点击测试，联通正常后才能保存" : "请填写完整后再测试")}
            </div>
              </>
            )}
          </div>

          <div className={`mt-3 rounded-2xl border p-3 shadow-sm transition-colors ${yoloEnvironmentCardClass}`}>
            <div className={`flex items-center justify-between ${isYoloEnvironmentCollapsed ? "" : "mb-2"}`}>
              <button
                type="button"
                onClick={() => setIsYoloEnvironmentCollapsed((collapsed) => !collapsed)}
                className="flex min-w-0 flex-1 items-center text-left focus:outline-none"
              >
                <span className="mr-1.5 shrink-0 text-slate-500">
                  {isYoloEnvironmentCollapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                </span>
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-slate-800">环境变量</div>
                  {isYoloEnvironmentCollapsed && (
                    <div className={`truncate text-[10px] ${isYoloEnvironmentTestPassed ? "text-emerald-700" : "text-red-600"}`}>
                      {hasYoloEnvironment ? yoloEnvironmentSubtitle : "本地训练运行环境"}
                    </div>
                  )}
                </div>
              </button>
              {!isYoloEnvironmentCollapsed && (
                <div className="flex items-center gap-1.5">
                <button
                  onClick={onTestYoloEnvironment}
                  disabled={
                    !hasYoloEnvironment ||
                    isTestingYoloEnvironment ||
                    isSavingYoloEnvironment
                  }
                  className="rounded-lg bg-indigo-600 px-2.5 py-1 text-xs font-medium text-white transition-colors hover:bg-indigo-700 disabled:bg-slate-300 disabled:text-slate-500"
                >
                  {isTestingYoloEnvironment ? "测试中" : "测试"}
                </button>
                <button
                  onClick={onSaveYoloEnvironment}
                  disabled={isSavingYoloEnvironment || isTestingYoloEnvironment}
                  className="rounded-lg bg-slate-900 px-2.5 py-1 text-xs font-medium text-white transition-colors hover:bg-slate-700 disabled:bg-slate-300 disabled:text-slate-500"
                >
                  {isSavingYoloEnvironment ? "保存中" : "保存"}
                </button>
              </div>
              )}
            </div>
            {!isYoloEnvironmentCollapsed && (
              <>
                <div className={`text-[10px] ${isYoloEnvironmentTestPassed ? "text-emerald-700" : "text-red-500"}`}>
                  {yoloEnvironmentSubtitle}
                </div>
                <label className="mt-2 block text-[11px] font-medium text-slate-500">
                  YOLO训练虚拟环境
                  <input
                    value={userSettings.local_yolo_train_venv_path}
                    placeholder="/path/to/venv 或 /path/to/venv/bin/yolo"
                    onChange={(event) =>
                      onUserSettingsChange({ ...userSettings, local_yolo_train_venv_path: event.target.value })
                    }
                    className={`${userSettingsInputClass(true)} placeholder:text-slate-300`}
                  />
                </label>
                <div className={`mt-2 text-[10px] ${isYoloEnvironmentTestPassed ? "text-emerald-700" : "text-red-600"}`}>
                  {yoloEnvironmentStatus ||
                    (hasYoloEnvironment
                      ? isYoloEnvironmentSaved
                        ? "已保存"
                        : "点击测试验证Ultralytics YOLO CLI是否可用"
                      : "未配置时，本地yaml训练会要求先填写该路径")}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
