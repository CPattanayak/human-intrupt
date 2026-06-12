from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langgraph.types import interrupt, Command
import uuid
import os

# -----------------------------
# State
# -----------------------------
class State(TypedDict):
    messages: List[dict]
    approved: bool

# -----------------------------
# LLM
# -----------------------------
llm =  ChatOpenAI(
    model="openai/gpt-4o-mini",  # Better for tool calling
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# -----------------------------
# Graph
# -----------------------------
graph = StateGraph(State)

async def draft_response(state: State):
    # Collect streamed chunks into a draft
    chunks = []
    async for chunk in llm.astream(state["messages"][-1]["content"]):
        chunks.append(chunk.content)
    draft_text = "".join(chunks)
    return Command(
        goto="review",
        update={"messages": state["messages"] + [{"role": "assistant", "content": draft_text}]}
    )

def human_review(state: State):
    # Interrupt here — ask human for approval
    return interrupt({"question": "Do you approve this draft? yes/no"})

def finalize(state: State):
    if state.get("approved"):
        return {"messages": state["messages"] + [{"role": "system", "content": "✅ Approved and delivered"}]}
    else:
        return {"messages": state["messages"] + [{"role": "system", "content": "❌ Rejected by human"}]}

graph.add_node("draft", draft_response)
graph.add_node("review", human_review)
graph.add_node("finalize", finalize)

graph.set_entry_point("draft")
graph.add_edge("draft", "review")
graph.add_edge("review", "finalize")
graph.set_finish_point("finalize")

memory = MemorySaver()
app_graph = graph.compile(checkpointer=memory)

# -----------------------------
# FastAPI with Swagger Metadata
# -----------------------------
api = FastAPI(
    title="LangGraph Human-in-the-Loop API",
    description="An API that streams draft responses from an LLM, pauses for human approval, and finalizes output.",
    version="1.0.0",
    contact={
        "name": "AI Engineering Team",
        "url": "https://example.com",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# -----------------------------
# Pydantic Models
# -----------------------------
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant').")
    content: str = Field(..., description="Content of the message.")

class Input(BaseModel):
    messages: List[Message] = Field(..., description="Conversation history as a list of messages.")
    approved: bool = Field(False, description="Human approval decision. True = approved, False = rejected.")
    thread_id: Optional[str] = Field(None, description="Unique identifier for the workflow session.")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "Write a professional email to my team about tomorrow's sprint planning."}
                ],
                "approved": False,
                "thread_id": "b8f1a4c2-9d3e-4f2a-9f7a-123456789abc"
            }
        }
class ApproveInput(BaseModel):
    approved: bool = Field(..., description="Human approval decision.")
    thread_id: str = Field(..., description="Workflow session ID to resume.")

# -----------------------------
# Endpoints
# -----------------------------
@api.post(
    "/chat/stream",
    summary="Stream draft response",
    description="Runs the graph to stream the draft response chunk-by-chunk, then pauses at the human review interrupt."
)
async def chat_stream(input: Input):
    thread_id = input.thread_id or str(uuid.uuid4())

    # Build initial state for the graph
    state = {
        "messages": [msg.dict() for msg in input.messages],
        "approved": False,
    }

    async def event_generator():
        # Stream through the graph itself, not just llm.astream
        async for chunk in app_graph.astream(
            state,
            config={"configurable": {"thread_id": thread_id}, "interrupt_before": ["review"]}
        ):
            # Each chunk is a partial update from the graph
            yield str(chunk)
        yield f"\n--- Awaiting human approval ---\n(thread_id={thread_id})\n"

    return StreamingResponse(event_generator(), media_type="text/plain")


@api.post(
    "/chat/approve",
    summary="Approve or reject draft",
    description="Resumes the workflow from the human review interrupt. Requires the same thread_id and an approval decision."
)
async def approve(input: ApproveInput):
    resume_value = {"approved": input.approved}
    result = await app_graph.ainvoke(
        Command(resume=resume_value),
        config={"configurable": {"thread_id": input.thread_id}, "resume_from": ["review"]}
    )
    return {"final": result["messages"]}

