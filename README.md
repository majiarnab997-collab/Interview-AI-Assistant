# 🎯 Interview AI Assistant

An AI-powered **mock interview platform** that combines multi-agent orchestration, retrieval-augmented generation (RAG), and automated evaluation to help candidates practice and improve their interview skills.

Built with **CrewAI**, **ChromaDB**, **Gradio**, and a **hybrid LLM architecture** (Groq + Gemini), the platform conducts realistic, persona-driven mock interviews grounded in the candidate's resume and target job description, then delivers actionable feedback and a final performance report.

---

## ✨ Features

- **Multi-Agent Interview Panel** — Powered by [CrewAI](https://github.com/crewAIInc/crewAI), with dedicated agents for evaluation, coaching, question generation, and final reporting.
- **Resume-Aware Questions (RAG)** — Upload a PDF resume; it's chunked, embedded, and stored in **ChromaDB** so interview questions and evaluations are grounded in your actual background.
- **Hybrid LLM Architecture**
  - **Groq** (`gpt-oss-120b`) — fast, low-latency tasks: coaching feedback delivery and RAGAS judge evaluation.
  - **Gemini** — deep reasoning tasks: technical evaluation, real-time company research (via Serper web search), and final interview audits.
- **Multiple Interviewer Personas**
  - Friendly HR
  - Strict Technical Interviewer
  - Behavioral Interviewer (STAR method)
- **Company-Aware Questioning** — Optionally provide a target company name; the question-generation agent researches its tech stack and engineering challenges to tailor questions.
- **Session Memory** — Candidate history is persisted locally (JSON) across sessions, so returning candidates get continuity from their previous performance.
- **RAGAS Evaluation** — Automated faithfulness and answer-relevancy scoring of the interview's Q&A flow using the [RAGAS](https://github.com/explodinggradients/ragas) framework.
- **Final Scorecard** — At the end of a session, an auditor agent produces a structured Markdown report: topics covered, strengths, weaknesses, and a final score out of 10.

> **Note:** A reference authentication module (PostgreSQL/NeonDB + bcrypt + email OTP) is included in the repo but intentionally **not wired into the live app**, so the public demo remains frictionless. See [Authentication Module](#-authentication-module-optional) below if you'd like to enable it.

---

## ✅ What's Implemented (Working in the Live App)

- [x] Gradio chat-based UI for conducting mock interviews
- [x] Resume PDF upload → text extraction → chunking → embeddings → ChromaDB storage (RAG)
- [x] Multi-agent interview round (Evaluator → Coach → Question Generator) via CrewAI
- [x] Hybrid LLM routing — Groq for fast tasks, Gemini for deep reasoning
- [x] Company-aware question generation using Serper web search
- [x] Three interviewer personas (Friendly HR, Strict Technical, Behavioral/STAR)
- [x] Turn-by-turn scoring (0–10) via structured LLM-as-a-judge output
- [x] End-of-interview summary report (topics, strengths, weaknesses, final score)
- [x] RAGAS evaluation (faithfulness, answer relevancy) on the retrieval log
- [x] Local JSON-based candidate session history across multiple interviews

## ❌ What's Not Implemented / Not Enabled

- [ ] **User authentication** — `auth.py` (bcrypt + PostgreSQL + email OTP) exists but is **not wired into `app.py`**; the live demo is open access with no login
- [ ] **Persistent database storage** — history is a local JSON file, which is wiped on restart on most free hosting tiers (no NeonDB/Postgres connection by default)
- [ ] **Multi-user isolation** — no per-user accounts, so all sessions share the same local ChromaDB collection and history file
- [ ] **Resume format support beyond PDF** — DOCX, TXT, or scanned/image-based resumes are not supported
- [ ] **Voice/audio interview mode** — text-only input and output, no speech-to-text or text-to-speech
- [ ] **Rate limiting / usage quotas** — no guardrails against hitting Groq/Gemini/Serper API limits
- [ ] **Automated tests / CI pipeline** — no test suite currently included

---

## 🏗️ Architecture

```
Resume PDF → Chunking → Embeddings → ChromaDB (RAG context)
                                          │
User Answer ──────────────────────────────┤
                                          ▼
                         ┌────────────────────────────┐
                         │   CrewAI: Interview Round   │
                         │                              │
                         │  1. Evaluator Agent (Gemini) │
                         │  2. Coach Agent (Groq)       │
                         │  3. Question Agent (Gemini)  │
                         └────────────────────────────┘
                                          │
                                          ▼
                         Chat Response + Next Question
                                          │
                              (repeat until interview ends)
                                          │
                                          ▼
                         ┌────────────────────────────┐
                         │  CrewAI: Summary Crew       │
                         │  Report Agent (Gemini)      │
                         └────────────────────────────┘
                                          │
                                          ▼
                     Final Scorecard + RAGAS Metrics + Saved History
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | [Gradio](https://gradio.app/) |
| Agent Orchestration | [CrewAI](https://github.com/crewAIInc/crewAI) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| PDF Parsing | `pypdf` |
| Text Chunking | `langchain-text-splitters` |
| Evaluation | [RAGAS](https://github.com/explodinggradients/ragas) |
| Fast LLM | Groq (`gpt-oss-120b`) |
| Reasoning LLM | Google Gemini |
| Web Search Tool | Serper.dev (via `crewai-tools`) |

---

## 📁 File Structure

```
interview-ai-assistant/
├── app.py           # Main application — Gradio UI, CrewAI agents, RAG, RAGAS eval
├── auth.py          # Reference auth module (NOT wired into app.py by default)
├── requirements.txt # Python dependencies
├── LICENSE          # MIT License
└── README.md        # This file
```

---

## 📋 Prerequisites

- Python 3.10+
- API keys for:
  - [Groq](https://console.groq.com/) → `GROQ_API_KEY`
  - [Google Gemini](https://ai.google.dev/) → `GEMINI_API_KEY`
  - [Serper.dev](https://serper.dev/) → `SERPER_API_KEY`

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/interview-ai-assistant.git
cd interview-ai-assistant
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file or export the following in your shell:

```bash
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
SERPER_API_KEY=your_serper_api_key
```

> On **Render**, add these under **Environment → Environment Variables** instead of a `.env` file (see [Deploying to Render](#️-deploying-to-render) below).

### 4. Run the app locally

```bash
python app.py
```

The app will start on `http://0.0.0.0:7860` by default (configurable via the `PORT` environment variable).

---

## ☁️ Deploying to Render

This app runs as a standard Python web service, which fits Render's **Web Service** deployment model.

1. **Push this repo to GitHub** (if you haven't already).

2. **Create a new Web Service on Render**
   - Go to [render.com](https://render.com) → **New +** → **Web Service**
   - Connect your GitHub repo

3. **Configure the service**

   | Setting | Value |
   |---|---|
   | Environment | `Python 3` |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `python app.py` |
   | Instance Type | At least **Standard** (the app loads embedding models and runs multi-agent LLM calls — the free tier's 512MB RAM is likely too tight) |

4. **Add environment variables** under **Environment → Environment Variables**:
   ```
   GROQ_API_KEY=your_groq_api_key
   GEMINI_API_KEY=your_gemini_api_key
   SERPER_API_KEY=your_serper_api_key
   ```
   Render automatically provides a `PORT` variable, which `app.py` already reads via `os.environ.get("PORT", 7860)` — no code changes needed.

5. **Deploy** — Render builds and starts the service, then gives you a public URL like `https://interview-ai-assistant.onrender.com`.

### ⚠️ Render-specific notes

- **Ephemeral disk**: Render's filesystem resets on redeploys/restarts, so `mock_interview_history.json` and the in-memory ChromaDB collection won't persist. For durable candidate history, add a database (e.g., Render's managed PostgreSQL) — this pairs naturally with `auth.py`'s existing `DATABASE_URL` pattern if you wire it in later.
- **Cold starts**: On lower-tier plans the service may spin down when idle and take a moment to restart on the next request.
- **Build time**: `sentence-transformers`, `torch`, and `chromadb` are heavy dependencies, so expect a longer first build.

---

## 🖥️ Usage

1. **Upload your resume** (PDF) and click **Process Resume**.
2. **Paste the job description** you're preparing for.
3. Optionally, enter a **target company name** so questions are tailored to that company's tech stack.
4. Enter a **candidate name/ID** to track your session history.
5. Choose an **interviewer persona** (Friendly HR, Strict Technical, or Behavioral).
6. Start typing your answers in the chat — the AI will evaluate each response, give coaching feedback, and ask a follow-up question.
7. Click **🎯 End Interview & Save Summary** to generate a final scorecard with strengths, weaknesses, and an overall score.

---

## 🔐 Authentication Module (Optional)

`auth.py` contains a production-ready authentication reference implementation using:

- PostgreSQL (NeonDB) for user storage
- `bcrypt` for password hashing
- Email-based OTP verification for registration

It is **not connected** to `app.py` by default. To enable it:

1. Set up a PostgreSQL database and create a `users` table with `username`, `email`, and `password_hash` columns.
2. Add these environment variables:
   ```bash
   DATABASE_URL=your_postgres_connection_string
   SENDER_EMAIL_ID=your_sender_email
   APP_PASSWORD_ID=your_email_app_password
   ```
3. Import the module in `app.py` and wire the login/register functions into the Gradio Blocks layout.

---

## ⚠️ Known Limitations

- **Ephemeral storage**: Candidate history is saved to a local JSON file, which does not persist across restarts/redeploys on Render. For persistent storage, connect a database (e.g., Render's managed PostgreSQL).
- **API rate limits**: Heavy use may hit rate limits on Groq, Gemini, or Serper free tiers.
- **No authentication by default**: The public demo runs without login. Enable the auth module above if you need access control.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [CrewAI](https://github.com/crewAIInc/crewAI) for multi-agent orchestration
- [RAGAS](https://github.com/explodinggradients/ragas) for RAG evaluation metrics
- [Gradio](https://gradio.app/) for the interactive UI