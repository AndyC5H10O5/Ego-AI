"""Agent 的嘴。一次成功调用对应一次终端发言。"""

SCHEMA = {
    "type": "function",
    "function": {
        "name": "speak",
        "description": "主动向用户说话。仅在你决定让外界听到这段话时调用。",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要对外说的原话"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}


def speak(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text 必须是非空字符串")
    print(text, flush=True)
    return "tool_calls=1"
