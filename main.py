import argparse
import codecs
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from queue import Empty, SimpleQueue
from threading import Thread
from typing import Optional

from openai import AuthenticationError, BadRequestError, OpenAI

from context_window import ROOT, ContextWindow, join_sections, load_fixed_prompt
from tool_calls import TOOL_SCHEMAS, ToolStep, execute_calls, read_assistant


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def load_dotenv_if_exists(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous internal thought with DeepSeek and optional live user input."
    )
    parser.add_argument(
        "--context-window", type=int, default=env_int("CONTEXT_WINDOW_TOKENS", 1048576),
        help="Total context window in tokens, including the response.",
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=env_int("MAX_OUTPUT_TOKENS", None),
        help="Tokens reserved for each response. Default: min(65536, window / 4).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=env_int("MAX_ROUNDS", None),
        help="Maximum rounds before exit. Default: unlimited.",
    )
    parser.add_argument(
        "--round-interval", type=float, default=os.getenv("ROUND_INTERVAL_SECONDS") or 0.5,
        help="Seconds to pause between internal thought rounds (default: %(default)s, 0 disables).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=env_int("MAX_CHARS", None),
        help="Stop if a round output exceeds this character count. Default: unlimited.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=env_int("MAX_RETRIES", 3),
        help="Retry times for API failure or empty output.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=env_float("RETRY_DELAY", 2.0),
        help="Seconds to wait between retries.",
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("LOG_FILE", ""),
        help="Log file path. Empty means auto-create one file per run.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.round_interval) or args.round_interval < 0:
        parser.error("--round-interval must be a finite, nonnegative number of seconds")
    return args


def get_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY. Please set it in env or .env file.")

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner").strip()
    if not model:
        raise ValueError("DEEPSEEK_MODEL cannot be empty.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def stream_once(client: OpenAI, model: str, messages: list[dict], output_tokens: int) -> dict:
    with client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        max_tokens=output_tokens,
        stream=True,
    ) as stream:
        return read_assistant(stream)


def prepare_log_file(log_file: str, model: str) -> str:
    path = log_file.strip()
    if not path:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("logs", f"run_{timestamp}.log")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write("=" * 72 + "\n")
        file.write(f"start_time={datetime.now().isoformat(timespec='seconds')}\n")
        file.write(f"model={model}\n")
        file.write("=" * 72 + "\n\n")
    return path


def append_log(
    log_file: str, round_index: int, answer: str, reasoning: str, elapsed: float, user_input: str,
    timestamp: str, tool_results: str = "",
) -> None:
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(
            f"[round={round_index} elapsed_s={elapsed:.2f} "
            f"answer_chars={len(answer)} reasoning_chars={len(reasoning)}]\n"
        )
        if user_input:
            file.write(user_input + "\n\n")
        if reasoning:
            file.write("[Thinking]\n")
            file.write(reasoning)
            file.write("\n\n")
        file.write(f"[Answer]\n{timestamp}\n")
        file.write(answer)
        file.write("\n\n")
        if tool_results:
            file.write(tool_results + "\n\n")


def read_user_input(inbox: SimpleQueue) -> None:
    stream = sys.stdin
    if stream is None:
        return
    if hasattr(stream, "reconfigure"):
        encoding = stream.encoding
        peek = getattr(getattr(stream, "buffer", None), "peek", None)
        if peek is not None and not stream.isatty():
            # Windows IDEs and redirected files can supply a BOM even when
            # Python has configured stdin as UTF-8. Peek without consuming text.
            prefix = peek(4)[:4]
            if prefix.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
                encoding = "utf-32"
            elif prefix.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
                encoding = "utf-16"
            elif prefix.startswith(codecs.BOM_UTF8):
                encoding = "utf-8-sig"
        stream.reconfigure(encoding=encoding, errors="replace")
    for line in stream:
        line = line.rstrip("\r\n")
        if line.replace("\ufffd", "").strip():
            inbox.put(line)


def take_user_input(inbox: SimpleQueue) -> str:
    lines = []
    while True:
        try:
            lines.append(inbox.get_nowait())
        except Empty:
            break
    return "【用户输入】\n" + "\n".join(lines) if lines else ""


def main() -> int:
    configure_output_encoding()
    load_dotenv_if_exists(ROOT / ".env")
    args = parse_args()

    try:
        client, model = get_client()
        output = args.max_output_tokens
        if output is None:
            output = min(65536, args.context_window // 4)
        tool_schema = json.dumps(TOOL_SCHEMAS, ensure_ascii=False)
        context = ContextWindow(load_fixed_prompt(), args.context_window, output, tool_schema)
        if args.max_retries < 0 or args.retry_delay < 0:
            raise ValueError("Retry count and delay must be nonnegative.")
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    history = ""
    latest = ""
    pending = ""
    steps: list[ToolStep] = []
    inbox = SimpleQueue()
    Thread(target=read_user_input, args=(inbox,), daemon=True).start()
    round_index = 1
    previous_round = round_index
    run_log_file = prepare_log_file(args.log_file, model)

    print("DeepSeek internal thought loop started. Type a line and press Enter to add input. Ctrl+C to stop.", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print(f"Log file: {os.path.abspath(run_log_file)}", file=sys.stderr)

    try:
        while True:
            if args.max_rounds is not None and round_index > args.max_rounds:
                print(f"\nReached max rounds ({args.max_rounds}). Exiting.", file=sys.stderr)
                break

            if round_index != previous_round and args.round_interval:
                time.sleep(args.round_interval)
            previous_round = round_index
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            clock_note = f"【当前时间】{timestamp}"
            clock_tokens = context.count("\n\n" + clock_note)
            attempt = 0
            assistant = None
            start_time = time.perf_counter()

            while attempt <= args.max_retries:
                try:
                    # Fold completed tool steps into ordinary memory only when the wire
                    # transcript itself would crowd out the next request.
                    reserved = sum(step.tokens for step in steps)
                    turn_limit = context.turn_limit - clock_tokens
                    while len(steps) > 1 and reserved > turn_limit - 128:
                        oldest = steps.pop(0)
                        history = join_sections(history, latest, oldest.memory)
                        latest = ""
                        reserved -= oldest.tokens
                    # A large final tool call still needs its receipt delivered. Borrow
                    # from the completion allowance before compacting that last step.
                    borrowed = max(0, reserved + 128 - turn_limit) if steps else 0
                    if steps and borrowed >= context.output:
                        history = join_sections(history, latest, steps.pop().memory)
                        latest = ""
                        reserved = borrowed = 0
                    output_tokens = context.output - borrowed

                    pending = join_sections(pending, take_user_input(inbox))
                    while True:
                        history, latest, pending = context.fit(
                            client, model, history, latest, pending, reserved - borrowed + clock_tokens
                        )
                        new_input = take_user_input(inbox)
                        if not new_input:
                            break
                        pending = join_sections(pending, new_input)

                    body = join_sections(history, latest)
                    if not steps:
                        body = join_sections(body, pending)
                    messages = context.messages(join_sections(body or "【脑内思考】", clock_note))
                    for step in steps:
                        messages.extend(step.messages)
                    if steps and pending:
                        messages.append({"role": "user", "content": pending})
                    assistant = stream_once(client, model, messages, output_tokens)
                    break
                except AuthenticationError:
                    print("DeepSeek authentication failed. Check DEEPSEEK_API_KEY.", file=sys.stderr)
                    return 1
                except BadRequestError as exc:
                    # Compatible endpoints can advertise a smaller window in their rejection.
                    match = re.search(r"(?:maximum|max) context length (?:is|of) ([\d,]+)", str(exc), re.I)
                    limit = int(match[1].replace(",", "")) if match else 0
                    if 0 < limit < context.window:
                        try:
                            context = ContextWindow(context.fixed, limit, min(context.output, limit // 4), tool_schema)
                        except ValueError as error:
                            print(f"Configuration error: {error}", file=sys.stderr)
                            return 1
                        print(f"Using server context window of {limit} tokens.", file=sys.stderr)
                        continue
                    print(f"Request rejected. Check model and token limits: {exc}", file=sys.stderr)
                    return 1
                except Exception as exc:
                    attempt += 1
                    if attempt > args.max_retries:
                        print(
                            f"\n[Round {round_index}] failed after retries: {exc}",
                            file=sys.stderr,
                        )
                        break
                    print(
                        f"\n[Round {round_index}] attempt {attempt}/{args.max_retries} failed: {exc}",
                        file=sys.stderr,
                    )
                    print(f"[Round {round_index}] retrying in {args.retry_delay:.1f}s...", file=sys.stderr)
                    time.sleep(args.retry_delay)

            if assistant is None:
                round_index += 1
                continue
            elapsed = time.perf_counter() - start_time
            round_output = assistant["content"]
            if args.max_chars is not None and len(round_output) > args.max_chars:
                print(f"[Round {round_index}] output exceeded max chars. Exiting.", file=sys.stderr)
                break

            tool_results = ""
            # Only the private answer gets the time prefix. Tool arguments and
            # character limits continue to use the model's original text.
            dated_answer = f"{timestamp}\n{round_output}" if round_output else timestamp
            if assistant.get("tool_calls"):
                # Only a fully received call can speak. A retry of the following request
                # reuses these results without executing the same calls again.
                assistant["content"] = dated_answer
                step = execute_calls(assistant, pending if steps else "", context.count)
                if not steps:
                    history = join_sections(history, latest, pending)
                    latest = ""
                steps.append(step)
                tool_results = step.results
            else:
                history = join_sections(history, latest, *(step.memory for step in steps), pending)
                latest = "【脑内思考】\n" + dated_answer
                steps.clear()
            append_log(run_log_file, round_index, round_output, assistant.get("reasoning_content", ""),
                       elapsed, pending, timestamp, tool_results)
            pending = ""
            if not assistant.get("tool_calls"):
                round_index += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user. Graceful shutdown complete.", file=sys.stderr)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
