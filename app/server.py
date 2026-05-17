from __future__ import annotations

import json
import os
import re
import hashlib
from collections.abc import Iterator
from typing import Any, Literal
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from app.agent.middleware import middleware_names
from app.agent.chat import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    OPENROUTER_MODEL,
    OPENROUTER_API_KEY,
    default_model_selection,
    get_chat_agent,
)
from app.agent.plan_execute import AgentPlan, run_plan_execute_agent, should_attempt_plan_execute, stream_plan_execute_agent
from app.agent.plan_execute.models import MissingField
from app.agent.plan_execute.supplement import parse_supplement
from app.agent.streaming import message_key, message_text
from app.agent.tool_registry import VISIBLE_TOOLS
from app.services.chat_store import init_chat_store, load_app_state_value, load_chat_state, save_app_state_value, save_chat_state
from app.services.user_settings import (
    load_user_settings,
    save_user_settings,
    test_user_settings_connection,
    test_yolo_environment,
)

load_dotenv()


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant", "model"]
    text: str = ""


class ChatRequest(BaseModel):
    provider: Literal["openrouter", "ollama"] | None = None
    model: str | None = None
    messages: list[ChatMessageIn] = Field(default_factory=list)


class ToolCallOut(BaseModel):
    id: str
    name: str
    args: dict[str, Any] | None = None
    status: Literal["done", "error"] = "done"
    result: Any | None = None


class ChatResponse(BaseModel):
    text: str
    toolCalls: list[ToolCallOut] = Field(default_factory=list)


class PersistedToolCall(BaseModel):
    id: str
    name: str
    args: Any | None = None
    status: str = "done"
    result: Any | None = None


class PersistedUIMessage(BaseModel):
    id: str
    role: Literal["user", "model"]
    text: str = ""
    toolCalls: list[PersistedToolCall] = Field(default_factory=list)


class PersistedChatSession(BaseModel):
    id: str
    name: str
    messages: list[PersistedUIMessage] = Field(default_factory=list)
    updatedAt: int


class PersistedChatState(BaseModel):
    sessions: list[PersistedChatSession] = Field(default_factory=list)
    currentSessionId: str = ""
    savedAt: int = 0


class UserSettings(BaseModel):
    remote_sftp_host: str = "172.31.1.42"
    remote_sftp_username: str = ""
    remote_sftp_private_key_path: str = "/home/qzq/.ssh/id_ed25519"
    remote_sftp_port: int = 22
    local_yolo_train_venv_path: str = ""


app = FastAPI(title="langCG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8765",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_chat_store()


def _split_env_list(name: str, fallback: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return [item for item in fallback if item]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _fetch_ollama_models() -> list[str]:
    if not OLLAMA_URL:
        return [OLLAMA_MODEL] if OLLAMA_MODEL else []

    base_url = OLLAMA_URL.rstrip("/")
    try:
        with urlopen(f"{base_url}/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return [OLLAMA_MODEL] if OLLAMA_MODEL else []

    models = payload.get("models")
    if not isinstance(models, list):
        return [OLLAMA_MODEL] if OLLAMA_MODEL else []

    names = []
    for item in models:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])

    if OLLAMA_MODEL and OLLAMA_MODEL not in names:
        names.insert(0, OLLAMA_MODEL)
    return names


def _to_langchain_messages(messages: list[ChatMessageIn]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        if not message.text:
            continue
        if message.role == "user":
            converted.append(HumanMessage(content=message.text))
        else:
            converted.append(AIMessage(content=message.text))
    return converted


def _last_user_text(messages: list[ChatMessageIn]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.text.strip():
            return message.text.strip()
    return ""


_LOCAL_PATH_ONLY_PATTERN = re.compile(r"^/[^\s，,。；;]+$")
_PENDING_PLAN_STATE_KEY = "pending_plan_clarifications"


def _conversation_key(messages: list[ChatMessageIn]) -> str:
    first_user_text = next((message.text.strip() for message in messages if message.role == "user" and message.text.strip()), "")
    current_user_count = sum(1 for message in messages if message.role == "user")
    seed = f"{first_user_text}|{max(0, current_user_count - 1)}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _load_pending_plans() -> dict[str, Any]:
    value = load_app_state_value(_PENDING_PLAN_STATE_KEY)
    pending = value.get("items")
    return pending if isinstance(pending, dict) else {}


def _save_pending_plans(pending: dict[str, Any]) -> None:
    save_app_state_value(_PENDING_PLAN_STATE_KEY, {"items": pending})


def _pending_key_for_next_turn(messages: list[ChatMessageIn]) -> str:
    return _conversation_key(messages)


def _pending_key_for_current_turn(messages: list[ChatMessageIn]) -> str:
    return _conversation_key(messages[:-1] if messages else messages)


def _save_pending_plan(messages: list[ChatMessageIn], request_text: str, plan: AgentPlan, clarification: Any) -> None:
    pending = _load_pending_plans()
    pending[_pending_key_for_next_turn(messages)] = {
        "request": request_text,
        "plan": plan.model_dump(mode="json"),
        "reason": clarification.render(plan) if hasattr(clarification, "render") else str(clarification),
        "missing_fields": [
            item.model_dump()
            for item in getattr(clarification, "missing_fields", [])
        ],
    }
    _save_pending_plans(pending)


def _pop_pending_plan(messages: list[ChatMessageIn]) -> dict[str, Any] | None:
    pending = _load_pending_plans()
    key = _pending_key_for_current_turn(messages)
    item = pending.pop(key, None)
    if item is not None:
        _save_pending_plans(pending)
    return item if isinstance(item, dict) else None


def _patch_pending_plan(item: dict[str, Any], user_text: str) -> tuple[str, AgentPlan] | None:
    try:
        plan = AgentPlan.model_validate(item.get("plan"))
    except Exception:
        return None

    missing_fields = [
        MissingField.from_dict(value)
        for value in item.get("missing_fields") or []
        if isinstance(value, dict)
    ]
    parsed = parse_supplement(user_text, missing_fields)
    if parsed.errors:
        return None

    patched = False
    for step in plan.steps:
        for missing in missing_fields:
            if missing.step_id != step.id or missing.tool != step.tool:
                continue
            if missing.field in parsed.values:
                step.args[missing.field] = parsed.values[missing.field]
                patched = True

    # Backward compatibility for pending plans created before missing_fields existed.
    if not missing_fields:
        for step in plan.steps:
            if step.tool != "publish_yolo_dataset":
                continue
            if _LOCAL_PATH_ONLY_PATTERN.match(user_text):
                step.args["detector_path"] = user_text
                patched = True
            elif "oldyaml" in user_text.lower():
                step.args["oldyaml"] = user_text.split("=", 1)[-1].strip()
                patched = True
            elif should_attempt_plan_execute(user_text):
                continue
            else:
                step.args["detector_path"] = user_text.strip()
                patched = True
    if not patched:
        return None

    original_request = str(item.get("request") or "")
    if _LOCAL_PATH_ONLY_PATTERN.match(user_text):
        request_text = f"{original_request}，发布到{user_text}"
    else:
        request_text = f"{original_request}，{user_text}"
    return request_text, plan


def _plan_execute_request_text(messages: list[ChatMessageIn]) -> str:
    latest_user_text = _last_user_text(messages)

    latest_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].role == "user" and messages[index].text.strip()
        ),
        None,
    )
    if latest_user_index is None:
        return latest_user_text

    if _LOCAL_PATH_ONLY_PATTERN.match(latest_user_text) or not should_attempt_plan_execute(latest_user_text):
        for index in range(latest_user_index - 1, -1, -1):
            message = messages[index]
            if message.role != "assistant" or "执行前需要补充信息" not in message.text:
                continue
            previous_user = next(
                (
                    item.text.strip()
                    for item in reversed(messages[:index])
                    if item.role == "user" and should_attempt_plan_execute(item.text.strip())
                ),
                "",
            )
            if not previous_user:
                return latest_user_text
            if _LOCAL_PATH_ONLY_PATTERN.match(latest_user_text):
                return f"{previous_user}，发布到{latest_user_text}"
            return f"{previous_user}，{latest_user_text}"

    if should_attempt_plan_execute(latest_user_text):
        return latest_user_text

    return latest_user_text


def _plan_execute_request_and_override(messages: list[ChatMessageIn]) -> tuple[str, AgentPlan | None]:
    latest_user_text = _last_user_text(messages)
    pending_item = _pop_pending_plan(messages)
    if pending_item is not None:
        patched = _patch_pending_plan(pending_item, latest_user_text)
        if patched is not None:
            return patched
    return _plan_execute_request_text(messages), None


def _run_plan_execute_if_matched(
    messages: list[ChatMessageIn],
    provider: str,
    model: str,
) -> ChatResponse | None:
    request_text, plan_override = _plan_execute_request_and_override(messages)
    if not should_attempt_plan_execute(request_text):
        return None
    result = run_plan_execute_agent(
        request=request_text,
        provider=provider,
        model=model,
        plan_override=plan_override,
    )
    if result is None:
        return None
    if result.fallback_reason:
        return ChatResponse(
            text=(
                "LangGraph plan/execute 未执行：计划生成或校验失败。\n"
                f"原因: `{result.fallback_reason}`\n\n"
                "该请求已识别为工具型任务，因此不会回退到旧 LangChain agent 自动执行。"
            ),
            toolCalls=[],
        )
    return ChatResponse(
        text=result.text,
        toolCalls=[
            ToolCallOut(
                id=call.id,
                name=call.name,
                args=call.args,
                status=call.status,
                result=call.result,
            )
            for call in result.tool_calls
        ],
    )


def _stream_plan_execute_if_matched(
    messages: list[ChatMessageIn],
    provider: str,
    model: str,
) -> Iterator[str] | None:
    request_text, plan_override = _plan_execute_request_and_override(messages)
    if not should_attempt_plan_execute(request_text):
        return None

    def _events() -> Iterator[str]:
        yield _sse("metadata", {"provider": provider, "model": f"{model} + langgraph-plan-execute"})
        yield _sse("progress", {"message": "已识别为工具型任务，进入 LangGraph plan/execute。"})
        yield _sse("progress", {"message": "正在规划步骤并校验参数..."})

        try:
            for event in stream_plan_execute_agent(
                request=request_text,
                provider=provider,
                model=model,
                plan_override=plan_override,
            ):
                event_type = event.get("type")
                if event_type == "fallback":
                    yield _sse(
                        "done",
                        {
                            "text": (
                                "LangGraph plan/execute 未执行：计划生成或校验失败。\n"
                                f"原因: `{event.get('reason')}`\n\n"
                                "该请求已识别为工具型任务，因此不会回退到旧 LangChain agent 自动执行。"
                            )
                        },
                    )
                    return
                if event_type == "plan":
                    plan = event["plan"]
                    lines = ["计划已生成，准备执行:"]
                    for index, step in enumerate(plan.steps, start=1):
                        lines.append(f"{index}. {step.id} -> {step.tool}")
                    yield _sse("progress", {"message": "\n".join(lines)})
                    continue
                if event_type == "clarification":
                    clarification = event["clarification"]
                    plan = event.get("plan")
                    if isinstance(plan, AgentPlan):
                        _save_pending_plan(messages, request_text, plan, clarification)
                    yield _sse("done", {"text": clarification.render(plan)})
                    return
                if event_type == "tool_start":
                    call = event["call"]
                    yield _sse("tool_call", {"id": call.id, "name": call.name, "args": call.args, "status": "running"})
                    continue
                if event_type == "tool_end":
                    call = event["call"]
                    yield _sse(
                        "tool_result",
                        {
                            "id": call.id,
                            "name": call.name,
                            "result": call.result,
                            "status": call.status,
                        },
                    )
                    continue
                if event_type == "done":
                    yield _sse("done", {"text": str(event.get("text") or "")})
                    return
        except Exception as error:
            yield _sse("error", {"message": str(error)})

    return _events()


def _run_agent(history: list[BaseMessage], provider: str, model: str) -> ChatResponse:
    agent = get_chat_agent(provider, model)
    emitted_messages: list[BaseMessage] = []
    seen_messages: set[tuple[Any, ...]] = set()

    for chunk in agent.stream(
        {"messages": history},
        stream_mode=["updates"],
        version="v2",
    ):
        if chunk["type"] != "updates":
            continue

        data = chunk.get("data") or {}
        if not isinstance(data, dict):
            continue

        for _step_name, step_data in data.items():
            if not isinstance(step_data, dict):
                continue
            for message in step_data.get("messages") or []:
                current_message_key = message_key(message)
                if current_message_key in seen_messages:
                    continue
                seen_messages.add(current_message_key)
                emitted_messages.append(message)

    tool_calls: dict[str, ToolCallOut] = {}
    response_parts: list[str] = []

    for message in emitted_messages:
        if isinstance(message, AIMessage):
            text = message_text(message.content)
            if text:
                response_parts.append(text)
            for call in message.tool_calls:
                call_id = str(call.get("id") or f"{call['name']}-{len(tool_calls)}")
                tool_calls[call_id] = ToolCallOut(
                    id=call_id,
                    name=call["name"],
                    args=call.get("args"),
                    status="done",
                )
        elif isinstance(message, ToolMessage):
            call_id = str(getattr(message, "tool_call_id", "") or getattr(message, "name", ""))
            result = message_text(message.content)
            if call_id and call_id in tool_calls:
                tool_calls[call_id].result = result
                continue
            if call_id:
                tool_calls[call_id] = ToolCallOut(
                    id=call_id,
                    name=getattr(message, "name", "tool"),
                    status="done",
                    result=result,
                )

    return ChatResponse(text="\n".join(response_parts).strip(), toolCalls=list(tool_calls.values()))


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _stream_agent_events(history: list[BaseMessage], provider: str, model: str) -> Iterator[str]:
    agent = get_chat_agent(provider, model)
    seen_messages: set[tuple[Any, ...]] = set()
    response_parts: list[str] = []

    yield _sse("metadata", {"provider": provider, "model": model})

    try:
        for chunk in agent.stream(
            {"messages": history},
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if chunk["type"] == "messages":
                token, metadata = chunk["data"]
                if not isinstance(metadata, dict) or metadata.get("langgraph_node") != "model":
                    continue
                text = message_text(token.content)
                if text:
                    response_parts.append(text)
                    yield _sse("token", {"text": text})
                continue

            if chunk["type"] != "updates":
                continue

            data = chunk.get("data") or {}
            if not isinstance(data, dict):
                continue

            for _step_name, step_data in data.items():
                if not isinstance(step_data, dict):
                    continue
                for message in step_data.get("messages") or []:
                    current_message_key = message_key(message)
                    if current_message_key in seen_messages:
                        continue
                    seen_messages.add(current_message_key)

                    if isinstance(message, AIMessage):
                        text = message_text(message.content)
                        if text and not response_parts:
                            response_parts.append(text)
                            yield _sse("token", {"text": text})
                        for index, call in enumerate(message.tool_calls):
                            call_id = str(call.get("id") or f"{call['name']}-{index}")
                            yield _sse(
                                "tool_call",
                                {
                                    "id": call_id,
                                    "name": call["name"],
                                    "args": call.get("args"),
                                    "status": "running",
                                },
                            )
                    elif isinstance(message, ToolMessage):
                        call_id = str(getattr(message, "tool_call_id", "") or getattr(message, "name", "tool"))
                        result = message_text(message.content)
                        yield _sse(
                            "tool_result",
                            {
                                "id": call_id,
                                "name": getattr(message, "name", "tool"),
                                "result": result,
                                "status": "error" if result.startswith("tool执行失败") else "done",
                            },
                        )

        yield _sse("done", {"text": "".join(response_parts).strip()})
    except Exception as error:
        yield _sse("error", {"message": str(error)})


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> dict[str, Any]:
    openrouter_models = (
        _split_env_list(
            "OPENROUTER_MODELS",
            [OPENROUTER_MODEL or "", "openrouter/auto", "google/gemini-2.5-flash"],
        )
        if OPENROUTER_API_KEY
        else []
    )
    ollama_models = _fetch_ollama_models()
    try:
        default_provider, default_model = default_model_selection()
    except RuntimeError:
        default_provider = "ollama" if OLLAMA_URL else "openrouter"
        default_model = (
            OLLAMA_MODEL
            or next(iter(ollama_models), "")
            or next(iter(openrouter_models), "")
        )

    return {
        "default": {"provider": default_provider, "model": default_model},
        "models": [
            {"provider": "openrouter", "model": model, "label": f"OpenRouter: {model}"}
            for model in openrouter_models
        ]
        + [
            {"provider": "ollama", "model": model, "label": f"Ollama: {model}"}
            for model in ollama_models
        ],
    }


@app.get("/api/tools")
def tools() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "callable": False,
                "execution": "langgraph_plan_execute_only",
                "parameters": tool.args_schema.model_json_schema()
                if tool.args_schema is not None
                else {"type": "object", "properties": {}},
            }
            for tool in VISIBLE_TOOLS.values()
        ]
    }


@app.get("/api/middleware")
def middleware() -> dict[str, Any]:
    return {"middleware": middleware_names()}


@app.get("/api/chat/state")
def get_chat_state() -> dict[str, Any]:
    try:
        return load_chat_state()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.put("/api/chat/state")
def put_chat_state(state: PersistedChatState) -> dict[str, str]:
    try:
        save_chat_state(state.model_dump(mode="json"))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {"status": "ok"}


@app.get("/api/user-settings")
def get_user_settings() -> dict[str, Any]:
    return load_user_settings()


@app.put("/api/user-settings")
def put_user_settings(settings: UserSettings) -> dict[str, Any]:
    try:
        return save_user_settings(settings.model_dump(mode="json"))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/user-settings/test")
def test_user_settings(settings: UserSettings) -> dict[str, Any]:
    try:
        return test_user_settings_connection(settings.model_dump(mode="json"))
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/user-settings/test-yolo-env")
def test_user_settings_yolo_env(settings: UserSettings) -> dict[str, Any]:
    try:
        return test_yolo_environment(settings.model_dump(mode="json"))
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        default_provider, default_model = default_model_selection()
        provider = request.provider or default_provider
        model = request.model or default_model
        plan_execute_response = _run_plan_execute_if_matched(request.messages, provider, model)
        if plan_execute_response is not None:
            return plan_execute_response

        history = _to_langchain_messages(request.messages)
        if not history:
            raise HTTPException(status_code=400, detail="messages 不能为空。")
        return _run_agent(history, provider, model)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    try:
        default_provider, default_model = default_model_selection()
        provider = request.provider or default_provider
        model = request.model or default_model
        plan_execute_events = _stream_plan_execute_if_matched(request.messages, provider, model)
        if plan_execute_events is not None:
            return StreamingResponse(
                plan_execute_events,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        history = _to_langchain_messages(request.messages)
        if not history:
            raise HTTPException(status_code=400, detail="messages 不能为空。")
        return StreamingResponse(
            _stream_agent_events(history, provider, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
