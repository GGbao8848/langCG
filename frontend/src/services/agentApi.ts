import { ChatModelOption, UIMessage } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `Request failed: ${response.status}`);
  }

  return response.json();
}

export async function fetchModels(): Promise<{
  default: { provider: string; model: string };
  models: ChatModelOption[];
}> {
  return requestJson("/api/models");
}

export async function fetchTools(): Promise<{ tools: any[] }> {
  return requestJson("/api/tools");
}

export async function sendChat(params: {
  provider: string;
  model: string;
  messages: UIMessage[];
}): Promise<Pick<UIMessage, "text" | "toolCalls">> {
  return requestJson("/api/chat", {
    method: "POST",
    body: JSON.stringify({
      provider: params.provider,
      model: params.model,
      messages: params.messages
        .filter((message) => message.text)
        .map((message) => ({
          role: message.role === "model" ? "assistant" : "user",
          text: message.text,
        })),
    }),
  });
}
