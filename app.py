"""
Interview AI Assistant — Mock Interview Platform
================================================
Production-ready application file for deployment on Hugging Face Spaces or local servers.
Combines Multi-Agent orchestration (CrewAI) + RAG (ChromaDB) + Session Persistence + RAGAS Evaluation.

Hybrid LLM Architecture:
  - Groq (fast, efficient): coaching feedback delivery & RAGAS assessment judging.
  - Gemini (deep reasoning): rigorous technical evaluation, real-time research, and final audits.
"""

import os
import json
from datetime import datetime


import gradio as gr
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

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
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# 1. Startup — Environment Variable Validation
# ============================================================
REQUIRED_ENV_VARS = ["GROQ_API_KEY", "GEMINI_API_KEY", "SERPER_API_KEY"]

missing_vars = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing_vars:
    raise EnvironmentError(
        f"❌ Missing required environment variables: {', '.join(missing_vars)}. "
        f"Please add them in Secrets (🔑 icon in the left sidebar)."
    )

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Local storage path for candidate session history (ephemeral on containerized free tiers)
HISTORY_FILE_PATH = "mock_interview_history.json"


# ============================================================
# 2. Hybrid LLM Setup
#    - Groq  → small/fast tasks (coaching feedback, RAGAS judging)
#    - Gemini → big/complex reasoning tasks (evaluation, question generation, reporting)
# ============================================================
groq_llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    max_tokens=800,
    temperature=0.7
)

gemini_llm = LLM(
    model="gemini/gemini-3.6-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.7
)

# RAGAS evaluation uses Groq (fast, mechanical statement-checking)
evaluator_llm = LangchainLLMWrapper(
    ChatGroq(model="openai/gpt-oss-120b", groq_api_key=GROQ_API_KEY)
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

search_tool = SerperDevTool()


# ============================================================
# 3. Structured Output Schema (LLM-as-a-Judge)
# ============================================================
class AnswerEvaluation(BaseModel):
    score: int = Field(ge=0, le=10, description="Numerical score from 0 to 10 evaluating technical depth and accuracy")
    technical_accuracy: str = Field(description="Evaluation of whether the candidate's answer is technically correct and grounded")
    key_strength: str = Field(description="The strongest architectural or technical point mentioned")
    key_gap: str = Field(description="Missing technical details, ambiguities, or weaknesses in the response")


# ============================================================
# 4. Agents (hybrid LLM assignment)
# ============================================================
evaluator_agent = Agent(
    role="Technical & Behavioral Answer Evaluator",
    goal="Critically analyze the candidate's answer for technical accuracy, depth, structure, and missing concepts without softening the critique.",
    backstory="You are a rigorous Principal Engineer and hiring assessment expert with over 15 years of technical interviewing experience. You dissect answers objectively, pinpoint exact flaws, verify claims against industry standards, and identify logical fallacies.",
    llm=gemini_llm,  # Big task: deep technical reasoning
    verbose=True
)

coach_agent = Agent(
    role="Interview Communication & Growth Coach",
    goal="Transform raw technical evaluation points into empathetic, highly actionable, and encouraging feedback for the candidate.",
    backstory="You are an experienced career mentor and communication coach. You excel at taking technical critiques and phrasing them constructively, highlighting what the candidate did well and offering one clear, concise tip for improvement.",
    llm=groq_llm,  # Small task: short paragraph rewrite
    verbose=True
)

question_generator_agent = Agent(
    role="Adaptive Technical Interviewer",
    goal="Generate one targeted, contextual interview question based on the job description, candidate resume, past weaknesses, and previous turn answers.",
    backstory="You are a seasoned engineering lead conducting structured interviews. You probe candidate claims, ask progressive follow-up questions to test system resilience and depth, and keep interview rounds engaging and relevant.",
    llm=gemini_llm,  # Big task: tool-use + complex context synthesis
    tools=[search_tool],
    verbose=True
)

report_agent = Agent(
    role="Comprehensive Mock Interview Auditor",
    goal="Synthesize the entire interview transcript into a structured final scorecard detailing topics covered, strengths, weaknesses, and a score out of 10.",
    backstory="You are an executive hiring bar raiser. You audit end-to-end interview records, correlate candidate responses with role requirements, and produce objective, data-backed evaluation summaries for hiring committees.",
    llm=gemini_llm,  # Big task: long-context synthesis
    verbose=True
)


# ============================================================
# 5. Tasks
# ============================================================
evaluation_task = Task(
    description="Evaluate the candidate's latest response: '{user_answer}'. Compare it against the recent question asked and the resume context: '{resume_context}'. Identify factual correctness, technical depth, missing architectural details, and any evasive or tangential statements without softening the feedback.",
    expected_output="A structured JSON-compliant evaluation containing a score from 0 to 10, technical_accuracy assessment, key_strength, and key_gap based on the candidate's answer.",
    agent=evaluator_agent,
    output_pydantic=AnswerEvaluation
)

coach_task = Task(
    description="Review the raw technical assessment from the evaluation task. Convert these findings into a constructive, encouraging, and human-like response tailored to the selected persona style: '{persona}'. Highlight one positive aspect and provide one concrete, actionable tip for improvement.",
    expected_output="A short, engaging paragraph (under 100 words) delivering constructive feedback directly to the candidate.",
    agent=coach_agent,
    context=[evaluation_task]
)

question_task = Task(
    description="Generate the next interview question for the candidate. Ground the question in the target company: '{company_name}', job description: '{job_description}', resume details: '{resume_context}', recent dialogue flow: '{conversation_history}', and previous evaluation gaps. If a company name is provided, use your search tool to research the company's tech stack, engineering challenges, or recent architecture practices to make the question highly realistic and role-specific. Ensure the question tests technical depth or problem-solving resilience.",
    expected_output="A single, highly focused, and challenging interview question tailored to the candidate's response. IMPORTANT: Return ONLY the plain text interview question in bold. Do NOT output raw JSON, tool call arguments, or search results.",
    agent=question_generator_agent
)

report_task = Task(
    description="Analyze the entire interview transcript: '{full_transcript}'. Audit all candidate responses against the target role requirements. Identify overall strengths, critical weaknesses, technical misconceptions, and calculate a realistic final score out of 10 with a clear rationale.",
    expected_output="A structured Markdown summary containing: ### Topics Covered, ### Candidate Strengths, ### Areas for Improvement / Weaknesses, and ### Score out of 10 with Rationale.",
    agent=report_agent
)


# ============================================================
# 6. Crews
# ============================================================
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
# 7. Vector DB (RAG)
# ============================================================
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="resume_pdf_db")


def process_resume(pdf_file):
    """Extracts text from the uploaded resume PDF, chunks it, and stores embeddings in ChromaDB."""
    global collection
    if not pdf_file:
        return "⚠️ Please upload a PDF file."

    try:
        chroma_client.delete_collection(name="resume_pdf_db")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name="resume_pdf_db")

    reader = PdfReader(pdf_file.name)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_text(full_text)

    resume_embeddings = embedding_model.encode(all_splits).tolist()
    chunk_ids = [f"chunk_{i}" for i in range(len(all_splits))]
    collection.upsert(documents=all_splits, embeddings=resume_embeddings, ids=chunk_ids)
    return "✅ Resume Processed Successfully!"


# ============================================================
# 8. Interviewer Personas
# ============================================================
personalities = {
    "Friendly HR": (
        "You are a warm, friendly, and encouraging HR interviewer conducting a mock interview. "
        "Your goal is to make the candidate feel comfortable while still evaluating their communication skills, "
        "confidence, and cultural fit. Ask one interview question at a time (behavioral or general HR-style, "
        "e.g. 'Tell me about yourself', 'Why do you want this role'). After the candidate answers, give brief, "
        "constructive, and positive feedback highlighting what they did well and one small area to improve. "
        "Keep your tone supportive and conversational, like a friendly senior colleague. "
        "Always end your response with the next interview question."
    ),
    "Strict Technical Interviewer": (
        "You are a strict, no-nonsense senior technical interviewer at a top tech company. "
        "Your goal is to rigorously evaluate the candidate's technical depth, problem-solving approach, and clarity of thought. "
        "Ask one technical question at a time (DSA, system design, or role-specific concepts depending on context). "
        "After the candidate answers, critically evaluate the correctness, depth, and structure of their answer. "
        "Point out gaps, incorrect assumptions, or missing edge cases directly and precisely, using formal technical terminology. "
        "Do not sugarcoat feedback, but remain professional and respectful. "
        "Always end your response with a follow-up question that increases in difficulty or probes deeper into their answer."
    ),
    "Behavioral Interviewer": (
        "You are a professional behavioral interviewer trained in the STAR method (Situation, Task, Action, Result). "
        "Your goal is to assess the candidate's soft skills, leadership, teamwork, and problem-solving through past experiences. "
        "Ask one behavioral question at a time (e.g. 'Tell me about a time you faced conflict in a team', "
        "'Describe a situation where you failed and what you learned'). "
        "After the candidate answers, evaluate whether their response followed a clear STAR structure, and give feedback "
        "on how they could make their answer more structured, specific, and impactful. "
        "Keep your tone calm, observant, and analytical. "
        "Always end your response with the next behavioral question."
    ),
}


# ============================================================
# 9. Long-Term Memory (JSON file — ephemeral on free hosting)
# ============================================================
def load_candidate_history():
    if os.path.exists(HISTORY_FILE_PATH):
        with open(HISTORY_FILE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_candidate_summary(candidate_id, new_session_data):
    history = load_candidate_history()
    history.setdefault(candidate_id, []).append(new_session_data)
    with open(HISTORY_FILE_PATH, "w") as f:
        json.dump(history, f, indent=4)


# ============================================================
# 10. Core Interview Logic
# ============================================================
def interview_bot(user_message, chat_history, job_description, persona,
                   conversation_history, candidate_id, company_name,
                   retrieval_log, session_scores):
    all_history = load_candidate_history()
    previous_history = ""
    candidate_sessions = all_history.get(candidate_id, [])

    if chat_history is None:
        chat_history = []
    if conversation_history is None:
        conversation_history = []
    if candidate_sessions:
        previous_history = candidate_sessions[-1].get("summary", "")

    query_vector = embedding_model.encode(user_message).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=2)
    context = "\n---\n".join(results["documents"][0]) if results["documents"] else ""

    try:
        interview_round_crew_result = interview_round_crew.kickoff(inputs={
            "user_answer": user_message,
            "resume_context": context,
            "persona": persona,
            "job_description": job_description,
            "conversation_history": conversation_history,
            "previous_history": previous_history,
            "company_name": company_name if company_name else "General Tech Company"
        })
    except Exception as e:
        error_reply = f"⚠️ Something went wrong while generating a response: {str(e)}"
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": error_reply})
        return "", chat_history, conversation_history, retrieval_log, session_scores

    coach_feedback = interview_round_crew_result.tasks_output[1].raw
    next_question = interview_round_crew_result.tasks_output[2].raw

    turn_score_obj = interview_round_crew_result.tasks_output[0].pydantic
    if turn_score_obj is not None and session_scores is not None:
        session_scores.append(turn_score_obj.score)

    bot_reply = f"{coach_feedback}\n\n{next_question}"

    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": bot_reply})
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_reply})

    retrieval_log.append({
        "question": user_message,
        "contexts": [context],
        "answer": next_question
    })

    return "", chat_history, conversation_history, retrieval_log, session_scores


def end_interview_and_save(conversation_history, persona, candidate_id, retrieval_log, session_scores):
    if len(conversation_history) == 0:
        return "⚠️ No interview history found to evaluate."

    full_transcript = ""
    for turn in conversation_history:
        full_transcript += f"{turn['role']}: {turn['content']}\n\n"

    avg_faithfulness = 0.0
    avg_relevancy = 0.0
    if retrieval_log:
        try:
            eval_dataset = Dataset.from_list(retrieval_log)
            result = evaluate(
                dataset=eval_dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings
            )
            faith_scores = result["faithfulness"]
            relev_scores = result["answer_relevancy"]
            avg_faithfulness = round(sum(faith_scores) / len(faith_scores), 2) if faith_scores else 0.0
            avg_relevancy = round(sum(relev_scores) / len(relev_scores), 2) if relev_scores else 0.0
        except Exception as e:
            print(f"⚠️ RAGAS evaluation failed: {e}")

    avg_score = round(sum(session_scores) / len(session_scores), 2) if session_scores else 0.0

    try:
        interview_summary_crew_result = interview_summary_crew.kickoff(inputs={"full_transcript": full_transcript})
        summary_bot_reply = interview_summary_crew_result.raw
    except Exception as e:
        summary_bot_reply = f"⚠️ Could not generate final report: {str(e)}"

    new_session_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "persona": persona,
        "summary": summary_bot_reply,
        "ragas_evaluation": {"faithfulness": avg_faithfulness, "answer_relevancy": avg_relevancy},
        "avg_llm_judge_score": avg_score
    }
    save_candidate_summary(candidate_id, new_session_data)
    return summary_bot_reply


# ============================================================
# 11. Gradio UI (Auth-free — Direct Access)
# ============================================================
with gr.Blocks(title="Interview AI Assistant") as demo:
    gr.Markdown("# 🎯 Interview AI Assistant — Mock Interview Platform")
    gr.Markdown(
        "Upload your resume, paste a job description, and practice with an AI-powered "
        "multi-agent mock interview panel. *(Demo mode — no login required.)*"
    )

    conversation_history = gr.State([])
    retrieval_log = gr.State([])
    session_scores = gr.State([])

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
            process_btn = gr.Button("Process Resume", variant="stop")
            status_box = gr.Textbox(placeholder="Current Status", label="Status", value="Waiting for resume...", interactive=False)
            jd_input = gr.Textbox(label="Job Description", lines=5, placeholder="Paste JD here...")
            company_input = gr.Textbox(
                label="Target Company Name",
                placeholder="e.g. Uber, Netflix, Swiggy, Razorpay...",
                value="Google",
            )
            candidate_input = gr.Textbox(label="Candidate Name / ID", value="Candidate_1", placeholder="Enter your name or ID...")
            end_interview_btn = gr.Button("🎯 End Interview & Save Summary", variant="primary")
            persona_radio = gr.Radio(
                choices=list(personalities.keys()),
                value="Friendly HR",
                label="Interviewer Persona"
            )
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
  server_port = int(os.environ.get("PORT", 7860))
  demo.launch(server_name="0.0.0.0", server_port=server_port)