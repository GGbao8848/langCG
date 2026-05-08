import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
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
).bind_tools(list(TOOLS.values()))


def _stream_ai_message(history: list) -> AIMessage:
    full_chunk: AIMessageChunk | None = None
    has_output = False

    for chunk in llm.stream(history):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            has_output = True
        full_chunk = chunk if full_chunk is None else full_chunk + chunk

    if has_output:
        print(flush=True)

    if full_chunk is None:
        return AIMessage(content="")

    return AIMessage(
        content=full_chunk.content,
        tool_calls=full_chunk.tool_calls,
        additional_kwargs=full_chunk.additional_kwargs,
        response_metadata=full_chunk.response_metadata,
        id=full_chunk.id,
    )


def _invoke_with_tools(history: list) -> AIMessage:
    while True:
        ai_message = _stream_ai_message(history)
        history.append(ai_message)

        if not ai_message.tool_calls:
            return ai_message

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool = TOOLS.get(tool_name)
            print(f"[tool] {tool_name}({tool_call['args']})", flush=True)
            if tool is None:
                tool_result = f"未找到tool: {tool_name}"
            else:
                try:
                    tool_result = tool.invoke(tool_call["args"])
                except Exception as exc:
                    tool_result = f"tool执行失败: {exc}"
            print(f"[tool-result] {tool_result}", flush=True)

            history.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )


def main() -> None:
    messages = [
        SystemMessage(
            content=(
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
        ),
        HumanMessage(content="你可以帮我做什么？"),
    ]

    _invoke_with_tools(messages)

    while True:
        user_input = input("Human: ")
        if user_input == "exit":
            print("再见！")
            break

        messages.append(HumanMessage(content=user_input))
        _invoke_with_tools(messages)


if __name__ == "__main__":
    main()
