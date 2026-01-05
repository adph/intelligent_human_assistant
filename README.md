# Intelligent Human Assistant — LLM Tooling & Automation 

An **Open WebUI**-hosted assistant that answers questions over your documents (RAG + OCR) **with citations** and can **take actions** by calling **MCP (FastMCP) tools** (e.g., run terminal commands or automate UI click/type steps).


---

## What it does

- **Document-grounded Q&A (RAG):** ingest PDFs/notes and answer questions with evidence-backed citations.
- **OCR-first ingestion:** extract text from scanned/low-quality PDFs and images to make them searchable.
- **Tool use via MCP:** the assistant can call tools such as:
  - `execute_command` – run a command in a constrained shell environment
  - `ui_click` / `ui_type` – coordinate-driven UI automation for repeatable workflows
- **Single chat surface:** everything runs through **Open WebUI**, so the user experience stays in one place.

---


## Tech stack

- **Python** for tool servers + ingestion pipeline
- **MCP (FastMCP)** for exposing tools to the chat runtime
- **Open WebUI** as the chat UI and tool host
- **RAG + OCR** for document ingestion and retrieval
- **UI automation** using coordinate-mapped actions (click/type) for deterministic workflows

---

## Repo layout (suggested)

Adjust to match your actual files—this is a recommended structure:

```
.
├── servers/
│   ├── terminal_server/          # MCP tool: execute_command
│   └── ui_automation_server/     # MCP tools: ui_click, ui_type
├── ingestion/
│   ├── ocr/                      # OCR adapters + preprocessing
│   └── rag/                      # chunking, embeddings, indexing
├── config/
│   ├── ui_maps/                  # JSON coordinate maps per app/screen
│   └── settings.example.yaml
├── scripts/
│   ├── ingest_docs.sh
│   └── run_all.sh
└── README.md
```

---

## Quickstart

### 1) Prerequisites

- Python **3.10+**
- Open WebUI running locally (Docker or native install)
- An LLM provider configured in Open WebUI (local or hosted)

### 2) Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### 3) Start MCP tool servers

Run each server in its own terminal (ports are examples—use yours).

```bash
# Terminal tool server
python -m servers.terminal_server --host 127.0.0.1 --port 8001

# UI automation tool server
python -m servers.ui_automation_server --host 127.0.0.1 --port 8002
```

### 4) Connect tools in Open WebUI

In Open WebUI:

1. Go to **Settings → Tools (MCP)** (wording may vary by version)
2. Add each MCP server endpoint (examples):
   - `http://127.0.0.1:8001`
   - `http://127.0.0.1:8002`
3. Refresh tools and verify you see:
   - `execute_command`
   - `ui_click`
   - `ui_type`

### 5) Ingest documents (RAG + OCR)

If you have an ingestion script:

```bash
bash scripts/ingest_docs.sh ./docs
```

Or run your Python entrypoint:

```bash
python -m ingestion.ingest --input ./docs --enable_ocr true
```

---

## Example prompts

Try these in Open WebUI:

- “Summarize the key points of **<document>** and cite the exact sections.”
- “Search my KB for references to **<term>**, then list the top 5 sources with citations.”
- “Open the browser and close the last tab.” *(requires UI map configured)*
- “Run `ls -la` in the project directory and paste the output.”

---

## UI automation notes (coordinate maps)

UI actions are deterministic when the environment is stable.

Recommended approach:
- Maintain `config/ui_maps/<app>.json` with named targets and coordinates.
- Version your maps per resolution / display scaling (e.g., `chrome_1728x1117@2x.json`).
- Include a “calibration” command or small script to regenerate coordinates.

Example map snippet:

```json
{
  "chrome": {
    "close_tab_button": {"x": 1685, "y": 72},
    "address_bar": {"x": 620, "y": 65}
  }
}
```

---

## Safety & guardrails

Because this project can run commands and click/type on your machine:

- Run tool servers **locally** and do not expose ports publicly.
- Use **allowlists** for commands if sharing with others.
- Consider a “dry-run” mode for UI actions.
- Log all tool calls for auditability.

---

## Roadmap (nice next steps)

- Replace coordinate-only UI control with element-based selectors where possible
- Add tool-call approvals (“Ask before running commands”)
- Add evaluation harness for retrieval quality (precision@k, citation faithfulness)
- Package as a single `docker compose` stack (Open WebUI + tool servers + vector DB)


