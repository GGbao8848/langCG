export type ToolCallData = {
  id: string; // unique internal ID
  name: string;
  args: any;
  status: "running" | "done" | "error";
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
