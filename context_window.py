import gzip
from pathlib import Path

from openai import OpenAI
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent


def load_fixed_prompt() -> str:
    sections = [
        path.read_text(encoding="utf-8-sig").strip()
        for path in sorted((ROOT / "prompts").glob("*.txt"))
    ]
    if not any(sections):
        raise ValueError("No fixed prompt found in prompts/*.txt.")
    return "\n\n".join(section for section in sections if section)


def join_sections(*sections: str) -> str:
    return "\n\n".join(section for section in sections if section)


class ContextWindow:
    def __init__(self, fixed: str, window: int, output: int):
        # Official offline tokenizer from cdn.deepseek.com/api-docs/deepseek_v4_tokenizer.zip
        with gzip.open(ROOT / "tokenizers_data/deepseek_v4.json.gz", "rt", encoding="utf-8") as f:
            self.tokenizer = Tokenizer.from_str(f.read())
        self.fixed = fixed
        self.output = output
        self.window = window
        self.memory_limit = window // 5
        self.summary_instruction = (
            "现在执行记忆压缩。下方材料按时间排列，可能包含已有摘要和新材料。"
            "用第一人称把它们合成一份可独立使用的记忆，区分用户原话和自己的判断。"
            "保留最重要的信息和未完成的思路，只输出完整摘要。"
        )
        # Both requests use exactly two messages. Leave room for chat template tokens.
        self.input_limit = window - self.count(fixed) - 32
        self.turn_limit = self.input_limit - output
        if output <= 0 or self.memory_limit <= 0 or min(
            self.turn_limit - self.memory_limit,
            self.input_limit - self.count(self.summary_instruction) - 2 * self.memory_limit,
        ) < 64:
            raise ValueError("Context window is too small for the fixed prompt, memory and output.")

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def messages(self, body: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.fixed}, {"role": "user", "content": body}]

    def summarize(self, client: OpenAI, model: str, source: str, limit: int) -> str:
        # Fold all source text into the summary, even when a single input exceeds the window.
        memory = ""
        offset = 0
        while offset < len(source):
            prefix = join_sections(
                self.summary_instruction, f"摘要须在 {limit} token 内写完。",
                "【已有摘要】\n" + memory if memory else "", "【本次材料】",
            )
            budget = self.input_limit - limit - self.count(prefix) - 4
            size = min(len(source) - offset, 6 * budget)
            while size > 0:
                body = join_sections(prefix, source[offset:offset + size])
                excess = self.count(body) + limit - self.input_limit
                if excess <= 0:
                    break
                size = min(size - 1, int(size * budget / (budget + excess) * 0.95))
            else:
                raise ValueError("No space for summary source text in the configured window.")
            result = client.chat.completions.create(
                model=model,
                messages=self.messages(body),
                max_tokens=limit,
                extra_body={"thinking": {"type": "disabled"}},
            )
            choice = result.choices[0]
            memory = (choice.message.content or "").strip()
            if choice.finish_reason != "stop" or not memory or self.count(memory) > limit:
                raise RuntimeError("Incomplete memory summary received. The source was retained.")
            offset += size
        return memory

    def fit(
        self, client: OpenAI, model: str, history: str, latest: str, incoming: str
    ) -> tuple[str, str, str]:
        if self.count(join_sections(history, latest, incoming)) <= self.turn_limit:
            return history, latest, incoming

        print("[Memory] Compressing context...", flush=True)
        # The retained previous output shares the 20% history budget with the summary.
        # New user input stays verbatim whenever it fits beside that memory.
        room = min(self.memory_limit, self.turn_limit - self.count(incoming) - 4)
        label = "【压缩记忆】\n"
        available = room - self.count(join_sections(label, latest)) - 4
        if available < 64:
            history, latest = join_sections(history, latest), ""
            available = room - self.count(label) - 4
        if available >= 64:
            history = label + self.summarize(client, model, history, available)
        else:
            # An exceptionally large user input also needs compression. Both summaries
            # share one memory budget and keep their different origins visible.
            remaining = self.memory_limit - self.count("【用户输入】\n") - 4
            if history:
                history = label + self.summarize(client, model, history, remaining // 2)
                remaining -= self.count(history) + 4
            incoming = "【用户输入】\n" + self.summarize(client, model, incoming, remaining)
        return history, latest, incoming
