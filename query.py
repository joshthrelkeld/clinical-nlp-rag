import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Load the existing vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Set up the LLM
llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0
)

# Custom prompt
prompt = PromptTemplate.from_template("""You are a research assistant helping users understand clinical NLP, ambient documentation, and physician burnout research.
Use the following context from the research papers to answer the question. If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query loop
print("Clinical NLP Assistant ready. Type 'quit' to exit.\n")
while True:
    question = input("Ask a question: ")
    if question.lower() == "quit":
        break
    result = chain.invoke(question)
    print(f"\nAnswer: {result}\n")