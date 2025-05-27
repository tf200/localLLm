from .vector_store import collection
import uuid
import json
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider



system_prompt = """
You are an expert assistant with access to a knowledge base.
Your job is to provide accurate, helpful answers based on the context provided in each query.
Format your response using Markdown for better readability (use headings, bold, italic, lists, code blocks, etc. as appropriate).
If the user's question is not covered by the provided context documents, say "I don't know."

IMPORTANT: At the end of your response, include a section titled "Sources Used" that lists the metadata 
of all the documents you referenced in your answer. This helps with attribution and transparency.
/no_think"""


# 4. Initialize your LLM agent
model = OpenAIModel(
    "gemma3-4B",
    provider=OpenAIProvider(base_url="http://localhost:8080/v1"),
)
agent = Agent(model=model, system_prompt=system_prompt)







async def stream_answer(question: str):
    # Retrieve top-5 docs
    results = collection.query(query_texts=[question], n_results=5)
    docs = results["documents"][0] # type: ignore
    metadata = results["metadatas"][0] # type: ignore

    context_with_metadata = []
    for i, (doc, meta) in enumerate(zip(docs, metadata)):
        doc_with_meta = f"DOCUMENT {i+1}:\n{doc}\nMETADATA: {meta}"
        context_with_metadata.append(doc_with_meta)

    context = "\n\n---\n\n".join(context_with_metadata)

    print(f"Context for question '{question}':\n{context}\n")

    user_prompt = f"""
Use the following retrieved documents to answer my question.

Context:
{context}

Question:
{question}

Remember to include a "Sources Used" section at the end of your response that lists the metadata of documents you referenced.
"""

    full_response = ""
    message_id = str(uuid.uuid4())  # You'll need to import uuid and generate this
    
    async with agent.run_stream(user_prompt) as result:
        async for delta in result.stream_text(delta=True):
            if delta:  # Only add non-empty deltas
                full_response += delta
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {
                            "delta": delta,
                            "message_id": message_id,
                            "chat_id": "current_chat_id",  # You'll need to replace this with actual chat_id
                        }
                    ),
                }
    
    # Send end event
    yield {
        "event": "done",
        "data": json.dumps({"text": ""})
    }