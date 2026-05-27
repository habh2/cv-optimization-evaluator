# CV Optimization evaluator — Project Context

## Project description

A multi-agent CV evaluator. Takes a job description and a master cv, and outputs detailed feedback. Human voice is enforced by a AI-phrase gate.

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
- Some resumes will be actually sent to their respective job applications which may result in no response, explicit rejection, or being contacted. I want this data to influence future runs so that the agents change their judging approach if I am not getting contacted.
- Observability should be a priority

