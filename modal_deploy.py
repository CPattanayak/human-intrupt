import modal

app = modal.App("human-in-the-loop-llm")

# Persistent volume for Hugging Face model cache
volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

# Build image with Hugging Face + FastAPI
image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi", "uvicorn", "langgraph", "langchain",
        "transformers", "torch", "peft", "email-validator"
    )
)

@app.function(
    image=image,
    gpu="T4",  # smaller GPU is enough for flan-t5-small
    volumes={"/model": volume} # mount volume at /model
    
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from pydantic import BaseModel, Field, ConfigDict
    from typing import List, Optional, TypedDict
    import uuid

    from langgraph.graph import StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt, Command

    # Load Hugging Face model from volume (downloaded on first run, cached thereafter)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir="/model")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", cache_dir="/model")

    # -----------------------------
    # State
    # -----------------------------
    class State(TypedDict):
        messages: List[dict]
        approved: bool

    graph = StateGraph(State)

    def draft_response(state: State):
        user_message = state["messages"][-1]["content"]
        inputs = tokenizer(user_message, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return Command(
            goto="review",
            update={"messages": state["messages"] + [{"role": "assistant", "content": response}]}
        )

    def human_review(state: State):
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
        title="Human-in-the-Loop LLM API",
        description=(
            "This API streams draft responses from a Hugging Face model, "
            "pauses for human approval, and finalizes output. "
            "Deployed on Modal with GPU acceleration and cached model volume."
        ),
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
    # Pydantic Models for Swagger
    # -----------------------------
    class Message(BaseModel):
        role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant').")
        content: str = Field(..., description="Content of the message.")

    class ChatInput(BaseModel):
        model_config = ConfigDict(json_schema_extra={
            "example": {
                "messages": [
                    {"role": "user", "content": "Write a professional email to my team about tomorrow's sprint planning."}
                ],
                "thread_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        })
        messages: List[Message] = Field(..., description="Conversation history as a list of messages.")
        thread_id: Optional[str] = Field(None, description="Unique identifier for the workflow session.")

    class ApproveInput(BaseModel):
        model_config = ConfigDict(json_schema_extra={
            "example": {
                "approved": True,
                "thread_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        })
        approved: bool = Field(..., description="Human approval decision.")
        thread_id: str = Field(..., description="Workflow session ID to resume.")

    class DraftResponse(BaseModel):
        draft: List[dict] = Field(..., description="Draft messages including assistant output.")
        thread_id: str = Field(..., description="Workflow session ID.")

    class FinalResponse(BaseModel):
        final: List[dict] = Field(..., description="Final messages after approval or rejection.")

    # -----------------------------
    # Endpoints
    # -----------------------------
    @api.post(
        "/chat",
        summary="Start chat and generate draft",
        description="Runs the graph to generate a draft response from the Hugging Face model. Pauses at human review interrupt.",
        response_model=DraftResponse,
        tags=["Chat"]
    )
    async def chat(input: ChatInput):
        thread_id = input.thread_id or str(uuid.uuid4())
        result = await app_graph.ainvoke(
            {"messages": [msg.model_dump() for msg in input.messages], "approved": False},
            config={"configurable": {"thread_id": thread_id}, "interrupt_before": ["review"]}
        )
        return {"draft": result["messages"], "thread_id": thread_id}

    @api.post(
        "/chat/approve",
        summary="Approve or reject draft",
        description="Resumes the workflow from the human review interrupt. Requires the same thread_id and an approval decision.",
        response_model=FinalResponse,
        tags=["Approval"]
    )
    async def approve(input: ApproveInput):
        # Resume with dict update that includes approved
        resume_value = {"approved": input.approved}
        result = await app_graph.ainvoke(
            Command(resume=resume_value),
            config={"configurable": {"thread_id": input.thread_id}, "resume_from": ["review"]}
        )
        return {"final": result["messages"]}

    return api
