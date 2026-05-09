import { AgentStreamEvent, ChatModelOption, UIMessage } from "../types";

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

export async function streamChat(
  params: {
    provider: string;
    model: string;
    messages: UIMessage[];
    signal?: AbortSignal;
  },
  onEvent: (event: AgentStreamEvent) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    signal: params.signal,
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

  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `Request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\n\n/);
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const event = parseSseBlock(block);
      if (event) onEvent(event);
    }
  }

  buffer += decoder.decode();
  const event = parseSseBlock(buffer);
  if (event) onEvent(event);
}

function parseSseBlock(block: string): AgentStreamEvent | null {
  const lines = block.split(/\n/).map((line) => line.trimEnd());
  const eventType = lines.find((line) => line.startsWith("event:"))?.slice("event:".length).trim();
  const data = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart())
    .join("\n");

  if (!eventType || !data) return null;

  return {
    type: eventType,
    ...JSON.parse(data),
  } as AgentStreamEvent;
}
