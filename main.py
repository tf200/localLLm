import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import chromadb
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

# 1. Initialize your Chroma client + collection
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection("Micla")

# 2. Define system prompt separately for clarity
system_prompt = """
You are an expert assistant with access to a knowledge base.
Your job is to provide accurate, helpful answers based on the context provided in each query.
Format your response using Markdown for better readability (use headings, bold, italic, lists, code blocks, etc. as appropriate).
If the user's question is not covered by the provided context documents, say "I don't know."
/no_think"""

# 3. Initialize your LLM agent with the system prompt
model = OpenAIModel(
    "Qwen3-4B",
    provider=OpenAIProvider(
        base_url="http://localhost:8080/v1",
    ),
)
agent = Agent(model=model, system_prompt=system_prompt)

# Create Rich console for beautiful terminal output
console = Console()


async def answer_question(question: str):
    # 4. Retrieve top‐5 docs for the question
    results = collection.query(
        query_texts=[question],
        n_results=5,
    )
    # Chroma returns lists of lists per query; we only queried one 
    docs = results["documents"][0]  # list of text strings

    # 5. Build your user prompt with context
    context = "\n\n---\n\n".join(docs)
    user_prompt = f"""
Use the following retrieved documents to answer my question.

Context:
{context}

Question:
{question}
"""

    # 6. Stream the response with real-time markdown rendering
    console.print("\n⏳ Thinking...\n")
    
    collected_text = ""
    
    console.print("\n")  # Add a bit of spacing
    with Live(Markdown(collected_text), refresh_per_second=10, console=console) as live:
        async with agent.run_stream(user_prompt) as result:
            async for chunk in result.stream_text(delta=True):
                collected_text += chunk
                # Update the Live display with latest markdown content
                live.update(Markdown(collected_text))
    
    console.print("\n")  # Add a bit of spacing after completion


async def main():
    console.print("🦙 [bold green]Welcome to your RAG terminal with Markdown support.[/bold green] Type 'exit' to quit.")
    while True:
        q = input("\nYour question ▶ ")
        if not q or q.lower() in {"exit", "quit"}:
            console.print("👋 [italic]Bye![/italic]")
            return
        await answer_question(q)


if __name__ == "__main__":
    asyncio.run(main())