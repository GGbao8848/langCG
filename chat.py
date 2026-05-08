import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from annotate_visualize_tool import annotate_visualize
from publish_yolo_dataset_tool import publish_yolo_dataset
from yolo_augment_tool import augment_yolo_dataset
from split_yolo_dataset_tool import split_yolo_dataset
from yolo_reindex_tool import reindex_yolo_labels
from xml_to_yolo_tool import convert_xml_to_yolo
from dataset_clean_tool import clean_irregular_dataset
from yolo_sliding_window_tool import yolo_sliding_window_crop


load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

SYSTEM_PROMPT = (
    "You are a helpful assistant for computer vision engineers. "
    "When the user asks to publish a YOLO dataset, incrementally publish with oldyaml, or publish to a detector path, "
    "use the publish_yolo_dataset tool. "
    "If oldyaml is provided, treat it as incremental publish. Otherwise, require a detector_path whose last path segment is detector_name. "
    "The tool itself decides whether the publish target is local or remote based on oldyaml or detector_path. "
    "When the user asks to visualize annotations, draw boxes, or render labels on images, "
    "use the annotate_visualize tool. "
    "If both YOLO and XML annotations exist for the same dataset unit, do not choose silently; ask the user whether to use yolo or xml. "
    "When the user asks to augment a YOLO dataset, "
    "use the augment_yolo_dataset tool. "
    "If the user says to use the default augmentation settings, call the tool directly without asking follow-up questions. "
    "The default augmentation settings are horizontal flip, vertical flip, brightness up, brightness down, contrast up, and contrast down enabled; left and right 90-degree rotation disabled unless the user explicitly asks for them. "
    "When the user asks to crop a YOLO dataset with a sliding window, "
    "use the yolo_sliding_window_crop tool. "
    "When the user asks to clean or reorganize an irregular dataset, "
    "use the clean_irregular_dataset tool. "
    "When the user asks to split a YOLO dataset into train only or train and val sets, "
    "use the split_yolo_dataset tool. "
    "When the user asks to convert Pascal VOC XML annotations into YOLO txt labels, "
    "use the convert_xml_to_yolo tool. "
    "When the user asks to remap YOLO class indices, such as converting 1 and 2 into 0 "
    "or applying a mapping like 0->0,1->0,2->3, "
    "use the reindex_yolo_labels tool."
)

TOOLS = {
    annotate_visualize.name: annotate_visualize,
    publish_yolo_dataset.name: publish_yolo_dataset,
    augment_yolo_dataset.name: augment_yolo_dataset,
    clean_irregular_dataset.name: clean_irregular_dataset,
    convert_xml_to_yolo.name: convert_xml_to_yolo,
    reindex_yolo_labels.name: reindex_yolo_labels,
    split_yolo_dataset.name: split_yolo_dataset,
    yolo_sliding_window_crop.name: yolo_sliding_window_crop,
}

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
    max_tokens=None,
)

agent = create_agent(
    model=llm,
    tools=list(TOOLS.values()),
    system_prompt=SYSTEM_PROMPT,
)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _message_key(message: BaseMessage) -> tuple[Any, ...]:
    tool_calls = getattr(message, "tool_calls", None)
    return (
        type(message).__name__,
        getattr(message, "id", None),
        _message_text(message.content),
        repr(tool_calls),
    )


def _print_tool_calls(message: AIMessage) -> None:
    for tool_call in message.tool_calls:
        print(f"[tool] {tool_call['name']}({tool_call['args']})", flush=True)


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
            text = _message_text(token.content)
            if text:
                print(text, end="", flush=True)
                has_output = True
            continue

        if chunk["type"] != "updates":
            continue

        for _step_name, step_data in chunk["data"].items():
            for message in step_data.get("messages", []):
                message_key = _message_key(message)
                if message_key in seen_messages:
                    continue
                seen_messages.add(message_key)
                emitted_messages.append(message)

                if isinstance(message, AIMessage) and message.tool_calls:
                    _print_tool_calls(message)
                elif isinstance(message, ToolMessage):
                    print(f"[tool-result] {_message_text(message.content)}", flush=True)

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
