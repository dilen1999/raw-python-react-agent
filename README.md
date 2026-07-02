# raw-python-react-agent

A **ReAct (Reasoning + Acting) agent loop implemented from scratch in Python**,
using the OpenAI API with strict JSON structured outputs — no LangChain,
no AutoGen, no CrewAI, no agent framework of any kind.

The goal of this project is to show exactly how agentic systems (and the
AI nodes inside tools like n8n or Make) actually work under the hood:
a loop that builds a prompt, calls an LLM, parses its structured reply,
executes a tool, appends the observation to history, and repeats.

## How the loop works

```
                ┌─────────────┐
                │  User Query │
                └──────┬──────┘
                       ▼
      ┌───────────────────────────────┐
      │   AGENT LOOP (agent.py)       │
      │                                │
      │  1. Build prompt               │◄────────────────┐
      │     (system prompt + history)  │                  │
      │  2. Call the LLM                │                  │
      │     (strict JSON schema)        │                  │
      │  3. Parse response              │                  │
      │     (parser.py)                 │                  │
      │        │                        │                  │
      │        ▼                        │                  │
      │  Malformed JSON? ───────────────┼── retry, up to   │
      │        │ no                     │   MAX_PARSE_RETRIES
      │        ▼                        │                  │
      │  action == final_answer? ──yes──┼──► return answer │
      │        │ no                     │                  │
      │        ▼                        │                  │
      │  action == human_handoff? ─yes──┼──► ask the user, │
      │        │ no                     │    feed answer   │
      │        ▼                        │    back in       │
      │  4. Execute tool                │                  │
      │     (tools.py registry)         │                  │
      │  5. Append observation          │──────────────────┘
      │  6. Loop (until MAX_ITERATIONS) │
      └───────────────────────────────┘
```

## Features

- **Agent loop from scratch** (`agent.py`) — plain Python `for` loop over
  build → call → parse → act → observe, with no framework in between.
- **Strict JSON structured outputs** (`prompts.py`) — the model is forced,
  via `response_format={"type": "json_schema", "strict": true, ...}`, to
  always reply with a `reasoning_summary`, an `action`, an `action_input`,
  and a `final_answer` field. This removes the need for fragile regex
  parsing of free-form "Thought / Action / Action Input" text.
- **Custom parser** (`parser.py`) — turns the raw model string into a
  validated dict, or raises a descriptive `AgentResponseError` covering:
  empty/`None` responses, responses wrapped in ` ```json ` fences,
  invalid JSON, missing required keys, unknown action names, and a
  `final_answer` action with no actual answer text.
- **Modular tool registry** (`tools.py`) — tools self-register with
  `@register_tool("name", "description")`. Adding a tool means writing
  one function and one decorator; the system prompt and JSON schema pick
  it up automatically via `describe_tools()`, so they can't drift out of
  sync with the code.
- **Human handoff** — the model can emit `action: "human_handoff"` with a
  `question` when it's missing information only a person can provide
  (an ambiguous request, a judgment call, or repeated tool failures).
  The loop pauses, asks the user directly (`input()` by default, or an
  injectable `input_fn` for testing), and feeds the answer back into the
  conversation as an observation.
- **Error-handling for malformed LLM responses** — parse failures are fed
  back to the model as an observation so it can self-correct, with a hard
  cap (`MAX_PARSE_RETRIES`) so a persistently broken model can't loop
  forever. API/network failures during the call itself are treated
  separately and fail fast, since retrying a self-correction message
  won't fix a dead connection.
- **Multi-turn conversation history** — every model turn and every
  observation is appended to a plain list of role/content dicts, which is
  exactly what gets replayed as the prompt on the next iteration.
- **Safety valves** — `MAX_ITERATIONS` caps the whole run, `MAX_PARSE_RETRIES`
  caps consecutive malformed replies, the calculator is AST-based (no
  `eval()`), and file tools are sandboxed to a `files/` directory with
  path-traversal checks.
- **Logging** — every step (reasoning, action, observation, parse errors)
  is written to `agent.log` in addition to stdout, for after-the-fact
  debugging of a run.
- **Unit tests** (`tests/`) — the parser, the tool registry, and the full
  agent loop (with the model call mocked out) are all covered, including
  the human-handoff path and both failure modes.

## Available tools

| Tool          | Purpose                                    | Input fields          |
|---------------|---------------------------------------------|------------------------|
| `calculator`  | Evaluate a math expression (AST-based, safe) | `expression`           |
| `web_search`  | Search the web for current information       | `query`                |
| `file_read`   | Read a file from the sandboxed `files/` dir   | `filename`             |
| `file_write`  | Write a file into the sandboxed `files/` dir  | `filename`, `content`  |

Plus two control-flow actions the loop handles directly: `human_handoff`
(`question`) and `final_answer`.

## Project structure

```
raw-python-react-agent/
├── agent.py           # the ReAct loop itself
├── main.py             # CLI entrypoint
├── parser.py            # validates/parses the model's structured JSON reply
├── prompts.py            # system prompt + JSON schema (generated from tools.py)
├── tools.py                # modular, self-registering tool implementations
├── files/                    # sandbox directory for file_read / file_write
├── tests/
│   ├── test_agent.py          # full loop, mocked model calls
│   ├── test_parser.py          # malformed-response handling
│   └── test_tools.py            # calculator + file tool behavior
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
git clone https://github.com/dilen1999/raw-python-react-agent.git
cd raw-python-react-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

## Usage

```bash
python main.py
```

```
ReAct Agent ready. Type 'exit' to quit.

You: what is 18 * 24, then write the result to result.txt
Agent: 18 * 24 is 432, and I've written it to result.txt.

You: exit
Done.
```

Every step of that run — reasoning, the `calculator` call, the observation,
the `file_write` call — is also written to `agent.log`.

## Running the tests

No API key or network access is required; the model call is mocked.

```bash
python -m unittest discover -s tests -v
```

## Design notes

- **Why no framework?** The point of this project is to make the agent
  loop legible: five plain steps, one Python class, no hidden prompt
  templates or callback machinery. Once you've built one from scratch,
  reading what LangGraph, n8n's AI Agent node, or CrewAI are doing under
  the hood stops feeling like magic.
- **Why structured outputs instead of "Thought:/Action:" text parsing?**
  Regex-based parsing of free-form ReAct text is one of the most common
  sources of agent flakiness. Forcing a JSON schema via `response_format`
  turns "did the model format its answer correctly" into a solved problem
  most of the time, and `parser.py` handles the remaining edge cases
  (fenced code blocks, truncated JSON, hallucinated action names)
  explicitly instead of silently swallowing them.
- **Why human handoff?** A ReAct agent that only ever calls tools or gives
  a (possibly wrong) final answer will confidently guess when it's
  missing information. Giving it an explicit `human_handoff` action makes
  "ask instead of guess" a first-class move in the loop, not an
  afterthought bolted onto the CLI.
