import os
import json
import anthropic
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
load_dotenv()

# Load vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Load paper metadata
with open("chroma_db/papers_metadata.json", "r") as f:
    papers_metadata = json.load(f)

# Initialize Anthropic client directly
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# --- Three functions ---

def search_all_papers(query: str, top_k: int = 8) -> str:
    """Semantic search across all papers. Returns chunks with source labels."""
    docs = vectorstore.similarity_search(query, k=top_k)
    results = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        results.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(results)


def search_specific_paper(paper_key: str, query: str, top_k: int = 6) -> str:
    """
    Semantic search filtered to one specific paper.
    Use when the question names a specific system or study.
    """
    meta = papers_metadata.get(paper_key)
    if not meta:
        return f"No paper found for key '{paper_key}'. Available keys: {list(papers_metadata.keys())}"

    filename = meta["filename"]

    # Filter ChromaDB results to only chunks from this paper
    docs = vectorstore.similarity_search(
        query,
        k=top_k,
        filter={"source": f"data/{filename}"}
    )

    if not docs:
        return f"No chunks found for {filename}. Check that the source metadata matches."

    return "\n\n".join(doc.page_content for doc in docs)


def get_paper_metadata(paper_key: str) -> str:
    """
    Returns structured metadata about a specific paper.
    Use for direct factual questions: what system, what methodology,
    what was the key finding, what was the focus area.
    """
    if paper_key == "all":
        return json.dumps(papers_metadata, indent=2)

    meta = papers_metadata.get(paper_key)
    if not meta:
        return f"No metadata found for '{paper_key}'. Available: {list(papers_metadata.keys())}"

    return json.dumps(meta, indent=2)


# --- Tool schemas ---

tools = [
    {
        "name": "search_all_papers",
        "description": """Semantic search across all seven clinical NLP papers.
                          Use for broad questions about ambient documentation,
                          physician burnout, or comparisons where the relevant
                          paper is not specified.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run across all papers"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve. Default 8.",
                    "default": 8
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_specific_paper",
        "description": """Semantic search filtered to one specific paper.
                          Use when the question explicitly names a system
                          like Abridge, Suki, or UCI Health, or when a
                          broad search returned too much noise.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_key": {
                    "type": "string",
                    "description": "The paper to search within. Must match a key in the metadata: abridge, suki, adt_burnout, nlp_eval_framework, benchmarking_ambient, tran_2023_nlcs, guo_2026_ambient_edits.",
                    "enum": list(papers_metadata.keys())
                },
                "query": {
                    "type": "string",
                    "description": "The search query to run within that paper"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve. Default 6.",
                    "default": 6
                }
            },
            "required": ["paper_key", "query"]
        }
    },
    {
        "name": "get_paper_metadata",
        "description": """Returns structured factual information about a
                          specific paper: system name, focus area, methodology,
                          and key finding. Use for direct factual lookups
                          rather than semantic search. Pass 'all' to see
                          metadata for every paper at once.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_key": {
                    "type": "string",
                    "description": """The paper key to retrieve metadata for,
                                      or 'all' for every paper."""
                }
            },
            "required": ["paper_key"]
        }
    }
]

available_functions = {
    "search_all_papers": search_all_papers,
    "search_specific_paper": search_specific_paper,
    "get_paper_metadata": get_paper_metadata
}

system_prompt = """You are a research assistant helping users understand
clinical NLP, ambient documentation, and physician burnout research across
seven published studies.

You have three tools:
- search_all_papers: broad semantic search across all papers
- search_specific_paper: targeted search within one named paper
- get_paper_metadata: direct factual lookup for a specific paper

Rules:
- Always use a tool before answering. Never answer from prior knowledge.
- If the question names a specific system (Abridge, Suki, etc.), use
  search_specific_paper rather than search_all_papers.
- If the question is purely factual (who, what methodology, what finding),
  use get_paper_metadata first.
- If tool results are insufficient, call a second tool before answering.
- If no relevant information is found after two tool calls, say so clearly."""


# --- Two-step execution loop ---

def answer_question(question: str) -> str:
    messages = [{"role": "user", "content": question}]

    # First call: model decides which tool to use
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system_prompt,
        tools=tools,
        messages=messages
    )

    # Execute tool calls
    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n[Tool called: {block.name}]")
                print(f"[Arguments: {block.input}]")
                fn = available_functions[block.name]
                result = fn(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "user", "content": tool_results})

        # Second call: synthesize into final answer
        final_response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=messages
        )

        # Handle case where model wants to call another tool
        if final_response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": final_response.content})

            tool_results = []
            for block in final_response.content:
                if block.type == "tool_use":
                    print(f"\n[Tool called: {block.name}]")
                    print(f"[Arguments: {block.input}]")
                    fn = available_functions[block.name]
                    result = fn(**block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})

            third_response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                system=system_prompt,
                tools=tools,
                messages=messages
            )
            return third_response.content[0].text

        return final_response.content[0].text

    return response.content[0].text

# Query loop
print("Clinical NLP Assistant ready. Type 'quit' to exit.\n")
while True:
    question = input("Ask a question: ")
    if question.lower() == "quit":
        break
    result = answer_question(question)
    print(f"\nAnswer: {result}\n")