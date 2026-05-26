from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from typing import TypedDict
import os


# -----------------------
# Graph State
# -----------------------
class State(TypedDict):
    jobdescription: str
    mastercv: str

    draft: str
    iteration: int

    jd_score: int
    humanstyle_score: int

    jd_feedback: str
    humanstyle_feedback: str

    avg_score: float


def get_llm(provider: str = "ollama", model: str = None):
    if provider == "ollama":
        return ChatOllama(model="qwen3:4b")

    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.5-flash",
            temperature=0.7
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


PROVIDER = "gemini"
MODEL = None

MAX_ITERATIONS = 4
TARGET_SCORE = 8.5


def loadfile(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
    print(f"Successfully read '{file_path}'")
    return data


def extract_text(response):
    """Normalize LLM response across Ollama, Gemini, OpenAI."""
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list):
            return content[0]["text"]
        if isinstance(content, str):
            return content
    return str(response)


llm = get_llm(PROVIDER)


# -----------------------
# Nodes
# -----------------------
def generate(state: State):
    if state["iteration"] == 0:
        prompt = f"""
Take the resume and add elements that would make it a better fit for the job description. Prioritize the critical requirements. Do not modify the job dates or titles. Don't add anything if the resume is already a good fit. You are allowed to remove items that might not help, are redundant, or duplicated. Do not provide meta-commentary.

JOB DESCRIPTION:
{state['jobdescription']}

RESUME:
{state['mastercv']}

"""
    else:
        prompt = f"""
Improve the resume draft based on the feedback. Just return plain text, so no emojis or markdown syntax. Do not modify the job dates or titles.

DRAFT:
{state['draft']}

JOB DESCRIPTION:
{state['jobdescription']}:

JOB DESCRIPTION FEEDBACK:
{state['jd_feedback']}

HUMAN STYLE FEEDBACK:
{state['humanstyle_feedback']}

"""

    response = llm.invoke(prompt)
    text = extract_text(response)
    return {
        "draft": text,
        "iteration": state["iteration"] + 1
    }


def humanstyle_eval(state: State):
    prompt = f"""
Evaluate the following resume to see if it was written by a human, as they might have generated it using AI. The content should still be professional, but deduct points for AI-generated phrasing or signs.

Return:
Score: int number from 1-10, dont' include the scale.
Feedback: short explanation of weaknesses and things to change.

Draft:
{state['draft']}

"""
    response = llm.invoke(prompt)
    text = extract_text(response)
    score = 5
    feedback = text

    for line in text.split("\n"):
        if "score" in line.lower():
            try:
                score = int("".join(filter(str.isdigit, line)))
                break
            except Exception:
                pass

    return {"humanstyle_score": score, "humanstyle_feedback": feedback}


def jd_eval(state: State):
    prompt = f"""
Evaluate the resume of a candidate against the job description.

Return:
Score: int number from 1-10, dont' include the scale.
Feedback: short explanation of weaknesses and things to change. If the match is too perfect to be true, include it as a weakness.

Draft:
{state['draft']}

Job description:
{state['jobdescription']}

"""
    r = llm.invoke(prompt)
    text = extract_text(r)

    score = 5
    for line in text.split("\n"):
        if "score" in line.lower():
            try:
                score = int("".join(filter(str.isdigit, line)))
            except Exception:
                pass

    return {"jd_score": score, "jd_feedback": text}


def aggregate(state: State):
    avg = (state["jd_score"] + state["humanstyle_score"]) / 2
    return {"avg_score": avg}


# -----------------------
# Router
# -----------------------
def should_continue(state: State):
    if state["avg_score"] >= TARGET_SCORE:
        return END
    if state["iteration"] >= MAX_ITERATIONS:
        return END
    return "generate"


# -----------------------
# Build Graph
# -----------------------
builder = StateGraph(State)

builder.add_node("generate", generate)
builder.add_node("jd_eval", jd_eval)
builder.add_node("humanstyle_eval", humanstyle_eval)
builder.add_node("aggregate", aggregate)

builder.set_entry_point("generate")

builder.add_edge("generate", "jd_eval")
builder.add_edge("generate", "humanstyle_eval")

builder.add_edge("jd_eval", "aggregate")
builder.add_edge("humanstyle_eval", "aggregate")

builder.add_conditional_edges(
    "aggregate",
    should_continue,
    {"generate": "generate", END: END}
)

graph = builder.compile()


# -----------------------
# Run  (temporary — will move to main.py in Week 1 refactor)
# -----------------------
if __name__ == "__main__":
    job_description = loadfile("data/inputs/JobDescription.txt")
    master_cv = loadfile("data/inputs/CV.md")

    result = graph.invoke({
        "mastercv": master_cv,
        "jobdescription": job_description,
        "draft": "",
        "score": 0,
        "iteration": 0
    })

    print("Final Draft:\n")
    print(result["draft"])
    print("\nScores:")
    print("JD score:", result["jd_score"])
    print("Human score:", result["humanstyle_score"])
    print("Average:", result["avg_score"])
    print("\nIterations:", result["iteration"])

    with open("data/outputs/ResultOutput.txt", "w", encoding="utf-8") as f:
        f.write(result["draft"])

    with open("data/outputs/ResultScores.txt", "w", encoding="utf-8") as f:
        f.write(f"JD score: {result['jd_score']}\n")
        f.write(f"Human score: {result['humanstyle_score']}\n")
        f.write(f"Average score: {result['avg_score']}\n")
        f.write(f"Iterations: {result['iteration']}\n -------------\n")
        f.write(f"JD FEEDBACK: \n{result['jd_feedback']}\n -------------\n")
        f.write(f"HUMAN STYLE FEEDBACK: \n{result['humanstyle_feedback']}\n -------------\n")
