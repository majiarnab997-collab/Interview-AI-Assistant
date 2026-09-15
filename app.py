"""
Interview AI Assistant — Mock Interview Platform (Memory-Optimized)
==================================================================
Lightweight build: Uses FastEmbed (ONNX) instead of heavy PyTorch/SentenceTransformers.
"""

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import json
from datetime import datetime

import gradio as gr
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from fastembed import TextEmbedding

from crewai import LLM, Task, Crew, Process, Agent
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

import sys, types
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _fake_chat_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
    class ChatVertexAI:
        pass
    _fake_chat_vertexai.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _fake_chat_vertexai

import langchain_community.llms as _lc_llms
if not hasattr(_lc_llms, "VertexAI"):
    class VertexAI:
        pass
    _lc_llms.VertexAI = VertexAI

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# ============================================================
# 1. Environment Variables Validation
# ============================================================
REQUIRED_ENV_VARS = ["GROQ_API_KEY", "GEMINI_API_KEY", "SERPER_API_KEY"]
missing_vars = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing_vars:
    raise EnvironmentError(f"❌ Missing environment variables: {', '.join(missing_vars)}")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HISTORY_FILE_PATH = "mock_interview_history.json"

# ============================================================
# 2. Hybrid LLM Setup & Lightweight Embeddings (FastEmbed)
# ============================================================
groq_llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    max_tokens=800,
    temperature=0.7
)

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.7
)

evaluator_llm = LangchainLLMWrapper(
    ChatGroq(model="openai/gpt-oss-120b", groq_api_key=GROQ_API_KEY)
)

# Ultra-lightweight ONNX embeddings (no PyTorch, ~120MB RAM)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
evaluator_embeddings = LangchainEmbeddingsWrapper(
    FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
)

search_tool = SerperDevTool()

# ============================================================
# 3. Structured Output Schema
# ============================================================
class AnswerEvaluation(BaseModel):
    score: int = Field(ge=0, le=10, description="Score 0 to 10")
    technical_accuracy: str = Field(description="Factual correctness")
    key_strength: str = Field(description="Strongest point")
    key_gap: str = Field(description="Missing points")

# ============================================================
# 4. Agents & Tasks
# ============================================================
evaluator_agent = Agent(
    role="Technical & Behavioral Evaluator",
    goal="Critique candidate answers rigorously.",
    backstory="Senior Principal Engineer conducting strict assessments.",
    llm=gemini_llm,
    verbose=True
)

coach_agent = Agent(
    role="Communication Coach",
    goal="Provide empathetic, actionable feedback.",
    backstory="Executive mentor simplifying critique into tips.",
    llm=groq_llm,
    verbose=True
)

question_generator_agent = Agent(
    role="Adaptive Interviewer",
    goal="Generate role-specific follow-ups.",
    backstory="Hiring manager adapting questions dynamically.",
    llm=gemini_llm,
    tools=[search_tool],
    verbose=True
)

report_agent = Agent(
    role="Interview Auditor",
    goal="Produce structured scorecard summaries.",
    backstory="Hiring bar-raiser evaluating cumulative performance.",
    llm=gemini_llm,
    verbose=True
)

evaluation_task = Task(
    description="Evaluate answer '{user_answer}' using context '{resume_context}'.",
    expected_output="JSON-compliant evaluation.",
    agent=evaluator_agent,
    output_pydantic=AnswerEvaluation
)

coach_task = Task(
    description="Convert evaluation into constructive feedback for persona '{persona}'.",
    expected_output="Paragraph under 100 words.",
    agent=coach_agent,
    context=[evaluation_task]
)

question_task = Task(
    description="Generate next question for company '{company_name}', JD: '{job_description}', Context: '{resume_context}'.",
    expected_output="Bold plain-text question.",
    agent=question_generator_agent
)

report_task = Task(
    description="Summarize transcript '{full_transcript}' with score out of 10.",
    expected_output="Markdown audit report.",
    agent=report_agent
)

interview_round_crew = Crew(
    agents=[evaluator_agent, coach_agent, question_generator_agent],
    tasks=[evaluation_task, coach_task, question_task],
    process=Process.sequential
)

interview_summary_crew = Crew(
    agents=[report_agent],
    tasks=[report_task],
    process=Process.sequential
)

# ============================================================
# 5. Vector DB (RAG)
# ============================================================
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="resume_pdf_db")

def process_resume(pdf_file):
    global collection
    if not pdf_file:
        return "⚠️ Please upload a PDF file."

    try:
        chroma_client.delete_collection(name="resume_pdf_db")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name="resume_pdf_db")

    reader = PdfReader(pdf_file.name)
    full_text = "".join([p.extract_text() or "" for p in reader.pages])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_text(full_text)

    # FastEmbed CPU inference
    resume_embeddings = [emb.tolist() for emb in embedding_model.embed(all_splits)]
    chunk_ids = [f"chunk_{i}" for i in range(len(all_splits))]
    collection.upsert(documents=all_splits, embeddings=resume_embeddings, ids=chunk_ids)
    return "✅ Resume Processed Successfully!"

personalities = {
    "Friendly HR": "Warm, supportive HR interviewer asking behavioral questions.",
    "Strict Technical Interviewer": "Direct, demanding senior engineer probing technical edge-cases.",
    "Behavioral Interviewer": "STAR method specialist assessing situational decisions."
}

def load_candidate_history():
    if os.path.exists(HISTORY_FILE_PATH):
        with open(HISTORY_FILE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_candidate_summary(candidate_id, data):
    history = load_candidate_history()
    history.setdefault(candidate_id, []).append(data)
    with open(HISTORY_FILE_PATH, "w") as f:
        json.dump(history, f, indent=4)

def interview_bot(user_message, chat_history, job_description, persona,
                   conversation_history, candidate_id, company_name,
                   retrieval_log, session_scores):
    chat_history = chat_history or []
    conversation_history = conversation_history or []
    candidate_sessions = load_candidate_history().get(candidate_id, [])
    previous_history = candidate_sessions[-1].get("summary", "") if candidate_sessions else ""

    # FastEmbed query vector
    query_vector = list(embedding_model.embed([user_message]))[0].tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=2)
    context = "\n---\n".join(results["documents"][0]) if results["documents"] else ""

    try:
        res = interview_round_crew.kickoff(inputs={
            "user_answer": user_message,
            "resume_context": context,
            "persona": persona,
            "job_description": job_description,
            "conversation_history": conversation_history,
            "previous_history": previous_history,
            "company_name": company_name or "General Tech Company"
        })
    except Exception as e:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": f"⚠️ Error: {str(e)}"})
        return "", chat_history, conversation_history, retrieval_log, session_scores

    coach_feedback = res.tasks_output[1].raw
    next_question = res.tasks_output[2].raw
    turn_score_obj = res.tasks_output[0].pydantic
    if turn_score_obj and session_scores is not None:
        session_scores.append(turn_score_obj.score)

    bot_reply = f"{coach_feedback}\n\n{next_question}"
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": bot_reply})
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_reply})

    retrieval_log.append({"question": user_message, "contexts": [context], "answer": next_question})
    return "", chat_history, conversation_history, retrieval_log, session_scores

def end_interview_and_save(conversation_history, persona, candidate_id, retrieval_log, session_scores):
    if not conversation_history:
        return "⚠️ No interview history found to evaluate."

    full_transcript = "".join([f"{t['role']}: {t['content']}\n\n" for t in conversation_history])
    avg_faithfulness, avg_relevancy = 0.0, 0.0

    if retrieval_log:
        try:
            eval_dataset = Dataset.from_list(retrieval_log)
            result = evaluate(
                dataset=eval_dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings
            )
            avg_faithfulness = round(sum(result["faithfulness"]) / len(result["faithfulness"]), 2)
            avg_relevancy = round(sum(result["answer_relevancy"]) / len(result["answer_relevancy"]), 2)
        except Exception as e:
            print(f"⚠️ RAGAS evaluation error: {e}")

    avg_score = round(sum(session_scores) / len(session_scores), 2) if session_scores else 0.0

    try:
        summary_bot_reply = interview_summary_crew.kickoff(inputs={"full_transcript": full_transcript}).raw
    except Exception as e:
        summary_bot_reply = f"⚠️ Could not generate final report: {str(e)}"

    save_candidate_summary(candidate_id, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "persona": persona,
        "summary": summary_bot_reply,
        "ragas_evaluation": {"faithfulness": avg_faithfulness, "answer_relevancy": avg_relevancy},
        "avg_llm_judge_score": avg_score
    })
    return summary_bot_reply

# ============================================================
# 6. Gradio UI
# ============================================================
with gr.Blocks(title="Interview AI Assistant") as demo:
    gr.Markdown("# 🎯 Interview AI Assistant — Mock Interview Platform")
    conversation_history = gr.State([])
    retrieval_log = gr.State([])
    session_scores = gr.State([])

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
            process_btn = gr.Button("Process Resume", variant="stop")
            status_box = gr.Textbox(label="Status", value="Waiting for resume...", interactive=False)
            jd_input = gr.Textbox(label="Job Description", lines=4, placeholder="Paste JD here...")
            company_input = gr.Textbox(label="Target Company Name", value="Google")
            candidate_input = gr.Textbox(label="Candidate ID", value="Candidate_1")
            end_interview_btn = gr.Button("🎯 End Interview & Save Summary", variant="primary")
            persona_radio = gr.Radio(choices=list(personalities.keys()), value="Friendly HR", label="Interviewer Persona")
        with gr.Column():
            chatbot = gr.Chatbot(height=500)
            msg_input = gr.Textbox(label="Your Answer", placeholder="Type here and press Enter...")
            gr.ClearButton([msg_input, chatbot, conversation_history])
            summary_output = gr.Markdown(label="Session Evaluation & Summary")

    process_btn.click(fn=process_resume, inputs=[pdf_input], outputs=[status_box])
    msg_input.submit(
        fn=interview_bot,
        inputs=[msg_input, chatbot, jd_input, persona_radio, conversation_history, candidate_input, company_input, retrieval_log, session_scores],
        outputs=[msg_input, chatbot, conversation_history, retrieval_log, session_scores]
    )
    end_interview_btn.click(
        fn=end_interview_and_save,
        inputs=[conversation_history, persona_radio, candidate_input, retrieval_log, session_scores],
        outputs=[summary_output]
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=False
    )