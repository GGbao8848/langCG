export type ToolCallData = {
  id: string; // unique internal ID
  name: string;
  args?: any;
  status: "running" | "done" | "error" | "canceled";
  result?: any;
};

export type UIMessage = {
  id: string;
  role: "user" | "model";
  text?: string;
  isProgress?: boolean;
  toolCalls?: ToolCallData[];
};

export type ChatSession = {
  id: string;
  name: string;
  messages: UIMessage[];
  updatedAt: number;
};

export type ChatModelOption = {
  provider: string;
  model: string;
  label: string;
};

export type LLMProvider = "openrouter" | "ollama";

export type LLMProviderSettings = {
  model: string;
  api_key: string;
  base_url: string;
};

export type LLMSettings = {
  active_provider: LLMProvider;
  providers: Record<LLMProvider, LLMProviderSettings>;
  model_options: Record<LLMProvider, string[]>;
};

export type UserSettings = {
  remote_sftp_host: string;
  remote_sftp_username: string;
  remote_sftp_private_key_path: string;
  remote_sftp_port: number;
  local_yolo_train_venv_path: string;
};

export type UserSettingsTestResult = {
  ok: boolean;
  message: string;
  latency_ms?: number;
};

export type AgentStreamEvent =
  | { type: "metadata"; provider: string; model: string }
  | { type: "progress"; message: string }
  | { type: "token"; text: string }
  | { type: "tool_call"; id: string; name: string; args?: any; status: "running" }
  | { type: "tool_result"; id: string; name?: string; result?: any; status: "done" | "error" }
  | { type: "done"; text: string }
  | { type: "error"; message: string };
