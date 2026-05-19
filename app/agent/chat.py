from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

from app.langchain_compat import apply_langchain_compatibility_patches

apply_langchain_compatibility_patches()

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from app.agent.middleware import build_agent_middleware
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.streaming import message_key, message_text, print_tool_calls, print_tool_result
from app.agent.tool_registry import CHAT_TOOLS
from app.services.llm_settings import load_llm_settings, validate_llm_settings


load_dotenv()


def default_model_selection() -> tuple[str, str]:
    settings = validate_llm_settings(load_llm_settings())
    return settings["provider"], settings["model"]


def _build_llm(
    provider: str | None = None,
    model: str | None = None,
) -> ChatOpenAI | ChatOllama:
    settings = validate_llm_settings(load_llm_settings())
    if provider is None or model is None:
        default_provider, default_model = settings["provider"], settings["model"]
        provider = provider or default_provider
        model = model or default_model
    if provider != settings["provider"]:
        raise RuntimeError("当前请求的LLM provider与已保存配置不一致，请在左下角“当前模型”中测试并保存后再试。")

    if provider == "openrouter":
        return ChatOpenAI(
            model=model,
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            temperature=0,
            max_tokens=None,
            stream_usage=False,
        )

    if provider == "ollama":
        return ChatOllama(
            model=model,
            base_url=settings["base_url"],
            temperature=0,
            num_ctx=8192,
            num_predict=2048,
            repeat_penalty=1.05,
            keep_alive="30m",
            client_kwargs={"trust_env": False, "timeout": 120},
            sync_client_kwargs={"trust_env": False, "timeout": 120},
            async_client_kwargs={"trust_env": False, "timeout": 120},
        )

    raise RuntimeError(f"不支持的 LLM provider: {provider}")


@lru_cache(maxsize=16)
def get_chat_agent(provider: str | None = None, model: str | None = None) -> Any:
    llm = _build_llm(provider, model)
    return create_agent(
        model=llm,
        tools=CHAT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=build_agent_middleware(
            summary_model=llm,
            tool_selector_model=llm,
        ),
    )


def _run_agent(history: list[BaseMessage]) -> list[BaseMessage]:
    agent = get_chat_agent()
    emitted_messages: list[BaseMessage] = []
    seen_messages: set[tuple[Any, ...]] = set()
    has_output = False

    for chunk in agent.stream(
        {"messages": history},
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, _metadata = chunk["data"]
            text = message_text(token.content)
            if text:
                print(text, end="", flush=True)
                has_output = True
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
                emitted_messages.append(message)

                if isinstance(message, AIMessage) and message.tool_calls:
                    print_tool_calls(message)
                elif isinstance(message, ToolMessage):
                    print_tool_result(message)

    if has_output:
        print(flush=True)

    return emitted_messages


def main() -> None:
    messages: list[BaseMessage] = [HumanMessage(content="你可以帮我做什么？")]
    messages.extend(_run_agent(messages))

    while True:
        user_input = input("Human: ")
        if user_input == "exit":
            print("再见！")
            break

        messages.append(HumanMessage(content=user_input))
        messages.extend(_run_agent(messages))


if __name__ == "__main__":
    main()
