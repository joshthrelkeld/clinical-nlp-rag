import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json
# Purely cosmetic. When LangChain releases the standalone package for PyPDF, this will be fixed
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# Load all PDFs from the data/ folder
data_dir = "data"
all_docs = []

for filename in os.listdir(data_dir):
    if filename.endswith(".pdf"):
        filepath = os.path.join(data_dir, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from {filename}")
        all_docs.extend(pages)

print(f"\nTotal pages loaded: {len(all_docs)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_docs)
# Verify source metadata is present on chunks
print(f"\nSample chunk metadata: {chunks[0].metadata}")
print(f"Split into {len(chunks)} chunks")

# Embed and store in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("\nVectorstore created and saved to chroma_db/")

papers_metadata = {
    "adt_burnout": {
        "filename": "ADT and Burnout Survey.pdf",
        "system": "Mass General Brigham / Emory Healthcare ADT Pilot",
        "focus": "ambient documentation technology, clinician burnout, physician wellbeing",
        "methodology": "Pre-post survey study across two academic medical centers, Professional Fulfillment Index, qualitative free-text analysis, REDCap",
        "key_finding": "ADT associated with 21.2% absolute reduction in burnout at MGB and 30.7% absolute increase in documentation-related wellbeing at Emory over 42-84 days"
    },
    "nlp_eval_framework": {
        "filename": "NLP Evaluation Framework 2024.pdf",
        "system": "CliniDeID, deid, MIST, NeuroNER, NLM Scrubber, Philter",
        "focus": "clinical NLP deidentification, PHI removal, evaluation methodology",
        "methodology": "Extensible shell-script pipeline using ETUDE evaluation tool across four corpora with partial, exact, and fully-contained annotation matching at multiple PII category granularities",
        "key_finding": "No single deidentification system uniformly outperformed others across corpora and PII categories; MIST was fastest at 24.57 notes/second, CliniDeID slowest at 1.00 note/second"
    },
    "benchmarking_ambient": {
        "filename": "Benchmarking Ambient Documentation.pdf",
        "system": "Multiple ambient scribe systems including Kaiser, TORTUS AI, DeepScribe, ACI-Bench",
        "focus": "evaluation frameworks, benchmarking metrics, ambient scribe assessment",
        "methodology": "Scoping review following PRISMA guidelines across PubMed, IEEE, Scopus, Web of Science, Embase; seven peer-reviewed studies included covering ROUGE, BERTScore, PDQI-9, SAIL, BLEURT metrics",
        "key_finding": "Wide heterogeneity in evaluation approaches with no standardized framework; only two datasets publicly available; ROUGE most common automated metric used in four of seven studies"
    },
    "tran_2023_nlcs": {
        "filename": "Tran 2023.pdf",
        "system": "Google Speech-to-Text Clinical, Amazon Transcribe Medical",
        "focus": "automatic speech recognition, non-lexical conversational sounds, ambient documentation accuracy",
        "methodology": "Quantitative and qualitative analysis of 36 primary care encounter transcripts re-enacted in professional audio studio; word error rate analysis with semantic typing of NLCS utterances",
        "key_finding": "ASR engines performed poorly on non-lexical conversational sounds with 94.7% error rate for Google and 98.7% for Amazon on clinically relevant NLCS like Mm-hm and Uh-uh, despite overall WER of only 11-12% for regular words"
    },
    "olson_2025_abridge_pharmacist": {
        "filename": "Abridge.pdf",
        "system": "Abridge (ambient documentation tool integrated with Epic Haiku, iOS)",
        "focus": "ambient clinical documentation, clinical pharmacists, medication therapy disease management, documentation burden, burnout, EHR efficiency, ambulatory care",
        "methodology": "Pre-post mixed-methods study at Geisinger health system; 30 MTDM pharmacists across 28 ambulatory clinics; Epic Signal EHR analytics for utilization rate and time-in-notes; pre/post Typeform surveys using NASA-TLX (documentation burden), Mini-Z (burnout), and Likert items; semi-structured interviews analyzed with Atlas.ti; LASSO variable selection + GEE regression; paired t-tests and Wilcoxon signed-rank tests; Cohen's d effect sizes",
        "key_finding": "ADT was activated for 65% of eligible encounters; average time spent in notes per encounter decreased by 86 seconds (-9.9%) post-ADT; documentation burden halved (Cohen's d: -1.90); undivided patient attention improved +25% (Cohen's d: 1.11); afterhours documentation reduced from 1.0 to 0.5 hours/week; burnout odds 56% lower post-ADT but non-significant (p=0.12); patient access unchanged; pharmacists consistently felt EHR time savings were underestimated by Epic Signal"
    },
    "palm_2025_suki_pdqi": {
        "filename": "Suki LLM.pdf",
        "system": "Suki AI (LLM-based ambient scribe integrated with EHR via ASR transcription pipeline)",
        "focus": "ambient clinical documentation quality, LLM note evaluation, hallucination detection, PDQI-9 validation, multi-specialty note comparison",
        "methodology": "Retrospective paired comparison of 97 de-identified outpatient audio encounters across 5 specialties (general medicine, pediatrics, OB/GYN, orthopedics, cardiology); LLM-generated 'Ambient' notes vs. blinded physician-authored 'Gold' notes; modified PDQI-9 (11 criteria on 1-5 Likert scale + binary hallucination detection); two blinded specialty-matched reviewers per note; interrater reliability via RWG statistic; paired t-tests and Mann-Whitney tests with Benjamini-Hochberg correction",
     "key_finding": "Ambient notes scored comparably to physician-authored Gold notes overall (4.20 vs. 4.25, p=0.04); Ambient notes outperformed on thoroughness (p<0.001) and organization (p=0.03); Gold notes outperformed on succinctness (p<0.001), accuracy (p=0.05), and internal consistency (p=0.004); hallucinations detected in 31% of Ambient vs. 20% of Gold notes (p=0.01); despite lower aggregate PDQI scores, reviewers preferred Ambient notes overall (47% vs. 39%); specialty-level preferences diverged, with OB/GYN and pediatrics favoring Gold and general medicine, orthopedics, and cardiology favoring Ambient"
    },
    "guo_2026_ambient_edits": {
        "filename": "From Conversation to Chart.pdf",
        "system": "Two commercial ambient AI documentation tools integrated with Epic EHR (Vendor A, Vendor B) at UCI Health",
        "focus": "clinician editing behavior, ambient AI documentation, note quality, turnaround time, EHR workflow",
        "methodology": "Retrospective analysis of 23,760 clinical notes (48,155 AI-drafted sections) from ambulatory clinics; Myers diff algorithm for token-level edit quantification; mixed-effects logistic regression for no-edit drivers; paired t-tests and ANOVA for linguistic metrics; logistic and mixed-effects regression for turnaround and edit intensity",
        "key_finding": "84.4% of AI-drafted notes were edited before signing; no-edit behavior was driven primarily by individual clinician style (ICC ~0.43–0.54) rather than specialty norms (ICC ≈ 0.00); editing was most intensive in Assessment & Plan; notes signed after 24 hours had lower edit intensity, suggesting delayed turnaround reflects workflow factors rather than heavier revision burden"
    }
}

with open("chroma_db/papers_metadata.json", "w") as f:
    json.dump(papers_metadata, f, indent=2)

print("Paper metadata saved to chroma_db/papers_metadata.json")