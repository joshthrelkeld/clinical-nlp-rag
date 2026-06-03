# Clinical NLP RAG Assistant

Seven peer-reviewed and preprint papers on ambient AI documentation + a LangChain, ChromaDB, and Claude RAG pipeline = a conversational tool for accessing clinical NLP research without digging through the literature yourself.

## Knowledge Base

| Paper | Focus |
|---|---|
| Tran et al. (2023), JAMIA | ASR performance and non-lexical conversational sounds in clinical settings |
| Suki LLM Evaluation (2025) | LLM-generated vs. specialist-drafted clinical notes across five specialties |
| From Conversation to Chart (2026) | Clinician edits to ambient AI draft notes at UCI Health |
| Abridge at Geisinger (2025) | Real-world deployment of Abridge with ambulatory pharmacists |
| Benchmarking Ambient Documentation (2025) | Systematic review of datasets and evaluation approaches |
| Heider & Meystre (2024) | Open-source evaluation framework for clinical NLP systems |
| ADT and Burnout Survey (2024) | Impact of ambient documentation technology on burnout across 1,430 clinicians |

## Stack

- **LangChain** — orchestration and LCEL chain
- **ChromaDB** — local vector store
- **HuggingFace sentence-transformers** — all-MiniLM-L6-v2 embeddings (local, free)
- **Anthropic Claude** — answer generation
- **pypdf** — PDF ingestion
- **Docker** — containerized deployment

## Setup

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/joshthrelkeld/clinical-nlp-rag.git
cd clinical-nlp-rag
```

Create a `.env` file in the project root:
ANTHROPIC_API_KEY=your_key_here

Add your PDFs to a `data/` folder, then run ingestion:
```bash
docker build -t clinical-nlp-rag .
docker run -it --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/chroma_db:/app/chroma_db clinical-nlp-rag python ingest.py
```

Start the query interface:
```bash
docker run -it --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/chroma_db:/app/chroma_db clinical-nlp-rag
```

### Option 2 — Local

```bash
git clone https://github.com/joshthrelkeld/clinical-nlp-rag.git
cd clinical-nlp-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
ANTHROPIC_API_KEY=your_key_here

Add your PDFs to a `data/` folder, then run ingestion:
```bash
python ingest.py
```

Start the query interface:
```bash
python query.py
```

## Example
Ask a question: How do BLEU and ROUGE metrics fall short for evaluating clinical NLP systems?
Answer: The context identifies a critical gap of limited integration of clinical
relevance in automated metrics. Studies suggest a shift is needed from traditional
NLP metrics like ROUGE and BERTScore to domain-specific measures such as clinical
accuracy and bias — traditional metrics don't adequately capture hallucinations,
internal consistency, or clinical comprehensibility.

## Limitations

Retrieval is chunk-based with a fixed size of 500 tokens. Dense tabular data and figure captions in PDFs may not retrieve cleanly. Increasing `k` in `query.py` improves coverage for complex queries.