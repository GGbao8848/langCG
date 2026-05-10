import { ChatSession } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
const DB_NAME = "langcg";
const DB_VERSION = 1;
const STORE_NAME = "appState";
const CHAT_STATE_KEY = "chat";
const LEGACY_SESSIONS_KEY = "langcg.sessions";
const LEGACY_CURRENT_SESSION_KEY = "langcg.currentSessionId";

export type PersistedChatState = {
  sessions: ChatSession[];
  currentSessionId: string;
  savedAt: number;
};

let pendingSave: Promise<void> = Promise.resolve();

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

async function readBackendState(): Promise<PersistedChatState | null> {
  const state = await requestJson<PersistedChatState>("/api/chat/state");
  return state.sessions.length > 0 ? state : null;
}

async function writeBackendState(state: PersistedChatState): Promise<void> {
  await requestJson("/api/chat/state", {
    method: "PUT",
    body: JSON.stringify(state),
  });
}

function openLegacyDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME);
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Failed to open legacy IndexedDB"));
  });
}

async function readLegacyIndexedDbState(): Promise<PersistedChatState | null> {
  const database = await openLegacyDatabase();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, "readonly");
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get(CHAT_STATE_KEY);

    request.onsuccess = () => {
      resolve(isPersistedChatState(request.result) && request.result.sessions.length > 0 ? request.result : null);
    };
    request.onerror = () => reject(request.error ?? new Error("Failed to read legacy IndexedDB"));
    transaction.oncomplete = () => database.close();
    transaction.onerror = () => database.close();
  });
}

async function clearLegacyIndexedDbState(): Promise<void> {
  const database = await openLegacyDatabase();

  return new Promise((resolve) => {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).delete(CHAT_STATE_KEY);
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      resolve();
    };
  });
}

function isPersistedChatState(value: unknown): value is PersistedChatState {
  if (!value || typeof value !== "object") return false;

  const candidate = value as Partial<PersistedChatState>;
  return (
    Array.isArray(candidate.sessions) &&
    typeof candidate.currentSessionId === "string" &&
    typeof candidate.savedAt === "number"
  );
}

function readLegacyLocalStorageState(): PersistedChatState | null {
  try {
    const rawSessions = localStorage.getItem(LEGACY_SESSIONS_KEY);
    if (!rawSessions) return null;

    const sessions = JSON.parse(rawSessions);
    if (!Array.isArray(sessions) || sessions.length === 0) return null;

    return {
      sessions,
      currentSessionId: localStorage.getItem(LEGACY_CURRENT_SESSION_KEY) ?? "",
      savedAt: Date.now(),
    };
  } catch (error) {
    console.warn("Failed to read legacy localStorage chat state", error);
    return null;
  }
}

function clearLegacyLocalStorageState() {
  localStorage.removeItem(LEGACY_SESSIONS_KEY);
  localStorage.removeItem(LEGACY_CURRENT_SESSION_KEY);
}

async function readLegacyState(): Promise<PersistedChatState | null> {
  try {
    return (await readLegacyIndexedDbState()) ?? readLegacyLocalStorageState();
  } catch (error) {
    console.warn("Failed to read legacy IndexedDB chat state", error);
    return readLegacyLocalStorageState();
  }
}

async function clearLegacyState(): Promise<void> {
  clearLegacyLocalStorageState();
  await clearLegacyIndexedDbState();
}

export async function loadPersistedChatState(): Promise<PersistedChatState | null> {
  try {
    const backendState = await readBackendState();
    if (backendState) return backendState;

    const legacyState = await readLegacyState();
    if (!legacyState) return null;

    await writeBackendState(legacyState);
    await clearLegacyState();
    return legacyState;
  } catch (error) {
    console.warn("Failed to load chat state from backend", error);
    return readLegacyState();
  }
}

export async function savePersistedChatState(state: PersistedChatState): Promise<void> {
  pendingSave = pendingSave
    .catch(() => undefined)
    .then(async () => {
      try {
        await writeBackendState(state);
      } catch (error) {
        console.warn("Failed to save chat state to backend", error);
        localStorage.setItem(LEGACY_SESSIONS_KEY, JSON.stringify(state.sessions));
        localStorage.setItem(LEGACY_CURRENT_SESSION_KEY, state.currentSessionId);
      }
    });

  return pendingSave;
}
