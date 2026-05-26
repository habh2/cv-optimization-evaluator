# CV Optimizer Agent — Project Context

## Project description

A multi-agent CV optimizer. Takes a job description + master CV, runs drafts through refinement agents, and outputs a single best-scoring CV. Human voice is enforced by a AI-phrase gate before LLM scoring.

## Porfolio goal

Demonstrates LangGraph multi-agent patterns, RAG, prod AI agent design, and LLM observability for AI Engineer / Applied AI roles. 

## LLM Instructions
 - Ask questions proactively. When asked to critique the project, focus on the high-level workflow while using the lens of a senior engineer (interviewer). Categorise the things to address by importance.
 - Related to the above, when coming up with a design plan or adjustment, think proactively what the critique would be and how to address it. 
 - This project is for a porfolio MVP, so keep the scope reasonable.
 - Architecture and design decisions must be defined in [DESIGN.md](DESIGN.md). Focus on documenting the high-level workflow, trade-offs, rationales and not things like file names, method classes or low level implementation details. 
 - Low-level details (state schema, agent interfaces, file structure) live in [INTERFACE.md](INTERFACE.md).

## Non-negotiable requirements:
- The design must align with ALL the portfolio goals since the objective is to showcase those skills.
- The objective of the optimizer is to produce a resume that scores better than the master resume for a given job description. The optimizer should also provide a better result than just passing the resume + the JD + instructions to a single LLM call (I am thinking of Claude skills focused on improving resumes).
- Some resumes will be actually sent to their respective job applications which may result in no response, explicit rejection, or being contacted. I want this data to influence future runs so that the agent know what works.
- Observability should answer the question: What changes where applied to improve this resume?


---

## Relationship to other projects

| Project | Connection |
|---------|-----------|

| `career-ops` | Natural pipeline: cv-optimizer-agent → `ResultCV.txt` → career-ops `/career-ops pdf` → ATS-clean PDF. Outcome tracking in career-ops `applications.md` feeds directly into `--outcome`/`--result` flags. Both tools read the same `cv.md` source via symlink. |

---

