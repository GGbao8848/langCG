import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.streaming import message_key, message_text, print_tool_calls, print_tool_result
from app.tools.annotate_visualize_tool import annotate_visualize
from app.tools.publish_yolo_dataset_tool import publish_yolo_dataset
from app.tools.yolo_augment_tool import augment_yolo_dataset
from app.tools.split_yolo_dataset_tool import split_yolo_dataset
from app.tools.yolo_reindex_tool import reindex_yolo_labels
from app.tools.xml_to_yolo_tool import convert_xml_to_yolo
from app.tools.dataset_clean_tool import clean_irregular_dataset
from app.tools.yolo_sliding_window_tool import yolo_sliding_window_crop


load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def _handle_tool_error(error: Exception) -> str:
    return f"tool执行失败: {error}"


def _make_safe_tool(tool: StructuredTool) -> StructuredTool:
    def _safe_func(**kwargs: Any) -> str:
        try:
            result = tool.invoke(kwargs)
        except Exception as error:
            return _handle_tool_error(error)
        return str(result)

    return StructuredTool.from_function(
        func=_safe_func,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        return_direct=tool.return_direct,
        response_format=tool.response_format,
    )

RAW_TOOLS = {
    annotate_visualize.name: annotate_visualize,
    publish_yolo_dataset.name: publish_yolo_dataset,
    augment_yolo_dataset.name: augment_yolo_dataset,
    clean_irregular_dataset.name: clean_irregular_dataset,
    convert_xml_to_yolo.name: convert_xml_to_yolo,
    reindex_yolo_labels.name: reindex_yolo_labels,
    split_yolo_dataset.name: split_yolo_dataset,
    yolo_sliding_window_crop.name: yolo_sliding_window_crop,
}

TOOLS = {name: _make_safe_tool(tool) for name, tool in RAW_TOOLS.items()}


def default_model_selection() -> tuple[str, str]:
    if OPENROUTER_API_KEY and OPENROUTER_MODEL:
        return "openrouter", OPENROUTER_MODEL
    if OLLAMA_MODEL and OLLAMA_URL:
        return "ollama", OLLAMA_MODEL
    raise RuntimeError(
        "未找到可用模型配置。请在 .env 中配置 OPENROUTER_API_KEY + OPENROUTER_MODEL，"
        "或配置 OLLAMA_URL + OLLAMA_MODEL。"
    )


def _build_llm(
    provider: str | None = None,
    model: str | None = None,
) -> ChatOpenAI | ChatOllama:
    if provider is None or model is None:
        default_provider, default_model = default_model_selection()
        provider = provider or default_provider
        model = model or default_model

    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY 未配置，无法使用 OpenRouter。")

        return ChatOpenAI(
            model=model,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
            max_tokens=None,
            stream_usage=False,
        )

    if provider == "ollama":
        if not OLLAMA_URL:
            raise RuntimeError("OLLAMA_URL 未配置，无法使用 Ollama。")

        return ChatOllama(
            model=model,
            base_url=OLLAMA_URL,
            temperature=0,
            max_tokens=None,
        )

    raise RuntimeError(f"不支持的 LLM provider: {provider}")


@lru_cache(maxsize=16)
def get_chat_agent(provider: str | None = None, model: str | None = None) -> Any:
    return create_agent(
        model=_build_llm(provider, model),
        tools=list(TOOLS.values()),
        system_prompt=SYSTEM_PROMPT,
    )


llm = _build_llm()

agent = get_chat_agent()


def _run_agent(history: list[BaseMessage]) -> list[BaseMessage]:
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

        for _step_name, step_data in chunk["data"].items():
            for message in step_data.get("messages", []):
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
