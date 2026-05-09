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

export type AgentStreamEvent =
  | { type: "metadata"; provider: string; model: string }
  | { type: "token"; text: string }
  | { type: "tool_call"; id: string; name: string; args?: any; status: "running" }
  | { type: "tool_result"; id: string; name?: string; result?: any; status: "done" | "error" }
  | { type: "done"; text: string }
  | { type: "error"; message: string };
