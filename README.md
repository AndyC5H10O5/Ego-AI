# Ego-AI

Ego-AI 是一个持续自我对话的命令行 Agent。启动后，模型自行生成第一段正文，程序再把正文接回下一轮。运行期间可以随时输入文字，模型也可以在没有用户输入的情况下继续。

## 在 Windows 上运行

在项目根目录打开 PowerShell。以下命令使用 Python 3.11 创建虚拟环境并安装依赖。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

首次配置时，从模板生成本机配置文件，然后在 `.env` 中填写 `DEEPSEEK_API_KEY`。已有同名环境变量时，程序优先使用环境变量。

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

可以先运行三轮，观察每轮的输出和日志。

```powershell
.\.venv\Scripts\python.exe main.py --max-rounds 3
```

持续运行时，使用 `.\.venv\Scripts\python.exe main.py`。输入一行文字并按 Enter，它会作为 `【用户输入】` 进入下一次尚未发出的对话请求。按 Ctrl+C 结束运行。每次启动都会开始一段新的自我对话，日志默认保存在当前工作目录的 `logs/` 下。

## 一轮对话如何运行

1. 启动时，程序按文件名顺序读取 `prompts/*.txt`，将它们合并为每次请求都携带的固定提示词。`prompts/00_self_awareness.txt` 描述循环和记忆机制。要添加固定目标，可以新建 `prompts/10_goals.txt`，以 `【目标】` 开头，重启后生效。
2. 每轮请求前，程序合并对话历史、上一轮正文和待处理的用户输入，用随项目附带的 DeepSeek 分词器计算 token 数，并为本轮输出预留空间。
3. 当剩余空间不足时，主循环请求模型将历史压缩为记忆，预算为上下文窗口的 20%。空间允许时保留上一轮正文，过长的材料会分块交给模型处理。
4. 模型的正文作为 `【自我对话】` 回送。接口返回的 `reasoning_content` 单独显示并记录为 `[Thinking]`，正文记录为 `[Answer]`。

## 配置入口

| 文件 | 用途 |
| --- | --- |
| `main.py` | 命令行入口、用户输入、流式输出、日志和重试 |
| `context_window.py` | 固定提示词加载、token 预算和记忆压缩 |
| `prompts/` | 自我意识及其他固定提示词 |
| `tokenizers_data/deepseek_v4.json.gz` | 运行时从本地读取的分词器数据 |
| `.env.example` | 可复制的配置模板，包含模型、接口地址、窗口大小和重试选项 |

在 `.env` 中用 `DEEPSEEK_MODEL` 指定模型，用 `CONTEXT_WINDOW_TOKENS` 设置总窗口，用 `MAX_OUTPUT_TOKENS` 设置单轮输出预算。运行 `.\.venv\Scripts\python.exe main.py --help` 可查看所有命令行参数。
