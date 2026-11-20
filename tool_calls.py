import json
from dataclasses import dataclass

from organ.speak import SCHEMA, speak


TOOLS = {SCHEMA["function"]["name"]: (SCHEMA, speak)}
TOOL_SCHEMAS = [schema for schema, _ in TOOLS.values()]


def read_assistant(stream) -> dict:
    """Assemble one complete assistant message before executing any side effects."""
    content = []
    reasoning = []
    calls = {}
    finish = None
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        finish = choice.finish_reason or finish
        delta = choice.delta
        if delta.content:
            content.append(delta.content)
        if thought := getattr(delta, "reasoning_content", None):
            reasoning.append(thought)
        for part in delta.tool_calls or ():
            call = calls.setdefault(part.index, {
                "type": "function", "id": "", "function": {"name": "", "arguments": ""},
            })
            call["id"] += part.id or ""
            if part.function:
                call["function"]["name"] += part.function.name or ""
                call["function"]["arguments"] += part.function.arguments or ""

    message = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if calls or finish == "tool_calls":
        ids = {call["id"] for call in calls.values()}
        if (finish != "tool_calls" or not calls or "" in ids or len(ids) != len(calls)
                or any(not call["function"]["name"] for call in calls.values())):
            raise RuntimeError("Incomplete tool-call stream.")
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    elif finish not in ("stop", "length") or (finish == "length" and not message["content"].strip()):
        raise RuntimeError("Incomplete self-dialogue stream.")
    return message


@dataclass
class ToolStep:
    messages: list[dict]
    memory: str
    tokens: int
    results: str


def execute_calls(assistant: dict, incoming: str, count_tokens) -> ToolStep:
    """Keep wire messages for the continuation and a readable record for future rounds."""
    messages = []
    memory = []
    results = []
    if incoming:
        messages.append({"role": "user", "content": incoming})
        memory.append(incoming)
    messages.append(assistant)
    if assistant["content"]:
        memory.append("【自我对话】\n" + assistant["content"])
    for call in assistant["tool_calls"]:
        name = call["function"]["name"]
        raw = call["function"]["arguments"]
        try:
            if name not in TOOLS:
                raise ValueError("未知工具")
            arguments = json.loads(raw)
            if not isinstance(arguments, dict):
                raise ValueError("参数必须是 JSON 对象")
            _, handler = TOOLS[name]
            result = handler(**arguments)
        except (TypeError, ValueError) as exc:
            result = f"调用失败：{exc}"
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        results.append(f"【工具记录】\n{name}({raw}) → {result}")
    record = "\n\n".join(results)
    memory.append(record)
    tokens = count_tokens(json.dumps(messages, ensure_ascii=False)) + 8 * len(messages)
    return ToolStep(messages, "\n\n".join(memory), tokens, record)
