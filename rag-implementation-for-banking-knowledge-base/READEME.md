### RAG Implementation for Banking Knowledge Base using LangChain

This response outlines a complete conceptual implementation of a Retrieval-Augmented Generation (RAG) system for a banking AI assistant using LangChain. The system handles loan products, regulatory requirements, and internal policies by processing a knowledge base of documents. I'll provide code snippets, explanations, and an architecture overview. In a real-world scenario, this would be hosted in a GitHub repository (e.g., structured with folders for `src/`, `docs/`, `data/`, and a `README.md` for setup instructions). The repo would include a `requirements.txt` with dependencies like `langchain`, `langchain-openai`, `chromadb`, `pypdf`, `unstructured`, etc.

Key focus areas:
- **Document Processing**: Handle complex tables in loan handbooks, regulatory manuals, policy documents, and rate sheets.
- **Challenges Addressed**: Prevent table context loss, handle cross-references, ensure consistent responses, and mitigate compliance risks via accurate retrieval.
- **LangChain Usage**: Leverage loaders, splitters, embeddings, vector stores, chains, and memory.
- **Cost Optimization**: Covered in a dedicated "Cost-Effective RAG Implementation Guide" section below.

#### Architecture Overview
The system follows a modular LangChain-based pipeline:

1. **Document Ingestion**: Load and preprocess documents using loaders like `PyPDFLoader` for PDFs and `UnstructuredFileLoader` for mixed formats.
2. **Chunking and Splitting**: Use a custom text splitter to preserve table structures (e.g., detect tables via metadata and split around them).
3. **Embeddings and Vector Storage**: Generate embeddings with OpenAI or HuggingFace models, store in Chroma (self-hosted for cost savings) or FAISS.
4. **Retrieval and Generation**: Use `RetrievalQA` chain for queries, with conversational memory via `ConversationalRetrievalChain`. Custom chains for banking workflows (e.g., compliance checks).
5. **Prompt Engineering**: Custom prompts to enforce regulatory accuracy and handle table data.
6. **Integration**: LLM (e.g., GPT-4 or local Llama) for generation.

**Architecture Diagram** (Text-based representation; in repo, use Mermaid or Draw.io for visual):

```
[User Query] --> [Conversational Memory] --> [Custom Prompt Template]
                                             |
                                             v
[Retrieval Chain] <-- [Vector Store (Chroma/FAISS)] <-- [Embeddings (OpenAI/HuggingFace)]
                  |
                  v
[Document Loader & Preprocessor] <-- [Knowledge Base Documents (PDFs, Docs)]
                  |
                  v
[Custom Text Splitter] (Preserves Tables & Cross-Refs)
```

This ensures end-to-end traceability: Queries retrieve chunks with intact table contexts, and responses cite sources for compliance.

#### Explanation of Custom Chunking Strategy for Tables
Standard chunking (e.g., `RecursiveCharacterTextSplitter`) often fragments tables, losing headers or row relationships. To address this:

- **Detection**: Use `unstructured` library to parse documents and extract tables as metadata-enriched chunks.
- **Preservation**: Create a custom splitter that identifies tables (via HTML-like tagging in unstructured output) and treats entire tables as atomic chunks if under max size. For large tables, split by rows but embed headers in each sub-chunk.
- **Cross-References**: During preprocessing, resolve "See Table X" by embedding reference links or duplicating referenced content in metadata.
- **Implementation**: Extend `RecursiveCharacterTextSplitter` to check for table markers and adjust splits.

This reduces inconsistent responses and compliance risks by ensuring retrieved chunks include full context (e.g., amortization table headers with data rows).

Example Code for Custom Chunking (in `src/chunking.py`):

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, NarrativeText

class TableAwareSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, documents):
        processed_docs = []
        for doc in documents:
            elements = partition_pdf(doc.page_content)  # Use unstructured to parse
            current_chunk = ""
            for element in elements:
                if isinstance(element, Table):
                    # Treat table as atomic if small, else split rows with header
                    table_text = str(element)
                    if len(table_text) > self.chunk_size:
                        header, rows = self._parse_table(table_text)
                        for row in rows:
                            sub_chunk = header + "\n" + row
                            processed_docs.append(sub_chunk)
                    else:
                        current_chunk += table_text + "\n"
                elif isinstance(element, NarrativeText):
                    current_chunk += str(element) + "\n"
                    if len(current_chunk) > self.chunk_size:
                        processed_docs.extend(super().split_text(current_chunk))
                        current_chunk = ""
            if current_chunk:
                processed_docs.extend(super().split_text(current_chunk))
        return processed_docs

    def _parse_table(self, table_text):
        # Custom logic to extract header and rows (e.g., split by lines)
        lines = table_text.split("\n")
        return lines[0], lines[1:]
```

This strategy maintains relationships, e.g., in regulatory matrices or pricing tables.

#### Main LangChain Implementation
Here's the core setup (in `src/main.py`). Assume environment variables for API keys.

```python
import os
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings  # Or HuggingFaceEmbeddings for cost savings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI  # Or from langchain_community.llms import Ollama for local

# Load documents
loaders = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredFileLoader
}
docs = []
for file in os.listdir("data/"):  # Knowledge base folder
    ext = os.path.splitext(file)[1]
    if ext in loaders:
        loader = loaders[ext](f"data/{file}")
        docs.extend(loader.load())

# Custom splitting
splitter = TableAwareSplitter(chunk_size=1500, chunk_overlap=300)
chunks = splitter.split_documents(docs)

# Embed and store
embeddings = OpenAIEmbeddings()  # Switch to HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2") for local
vectorstore = Chroma.from_documents(chunks, embeddings)

# Custom prompt for banking compliance
prompt_template = """
You are a banking AI assistant. Answer based on the context, ensuring regulatory compliance. If unsure, say "Consult a human expert."
Context: {context}
Question: {question}
Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Retrieval chain with memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name="gpt-4"),  # Or Ollama(model="llama3")
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

# Example query
response = qa_chain({"question": "What are the amortization terms for personal loans?"})
print(response["answer"])

# Custom chain for workflows (e.g., loan eligibility check)
from langchain.chains import LLMChain
eligibility_chain = LLMChain(llm=OpenAI(), prompt=PromptTemplate.from_template("Check eligibility for {loan_type} based on {user_data}."))
full_chain = qa_chain | eligibility_chain  # Chain orchestration
```

This demonstrates loaders, splitters, vector stores, embeddings, chains, memory, and custom prompts. For conversation context, memory tracks history to avoid redundant retrievals.

In the repo, add tests (e.g., unit tests for chunking) and a Streamlit app for demo UI.

### Cost-Effective RAG Implementation Guide

This supplementary guide analyzes costs for enterprise RAG in banking, focusing on high-cost components, alternatives, breakdowns, trade-offs, recommendations, and ROI.

#### High-Cost Components to Optimize
- **Premium LLM APIs (e.g., GPT-4)**: Token-based pricing drives 70-80% of costs in query-heavy systems.
<argument name="citation_id">36</argument>
 Real-time generation for complex queries (e.g., table interpretation) amplifies this.
- **Vector Database Hosting (e.g., Pinecone)**: Cloud fees for storage and queries, scaling with document volume.
<argument name="citation_id">35</argument>

- **Document Processing APIs (e.g., Unstructured.io)**: Per-page fees for advanced table extraction.
- **Compute for Embeddings**: GPU-intensive, costly on cloud.

#### Cost-Effective Alternatives
- **Local/Open-Source LLMs**: Use Ollama with Llama 3 or Mistral to eliminate API costs. Self-host on local hardware for privacy/compliance.
<argument name="citation_id">5</argument>

<argument name="citation_id">8</argument>

- **Self-Hosted Vector DBs**: Chroma or FAISS (free, open-source) on your servers; no hosting fees beyond infra.
<argument name="citation_id">0</argument>

- **Batch Processing**: Embed documents offline during ingestion, not per query.
- **Embedding Caching**: Store pre-computed embeddings in local files to avoid recomputation.
- **Tiered LLM Strategy**: Route simple queries (e.g., policy lookups) to cheaper models like GPT-4.1 mini ($0.40/1M input tokens) or local models; complex ones (e.g., regulatory analysis) to premium.
<argument name="citation_id">36</argument>

<argument name="citation_id">25</argument>
 Additional strategies: Prompt compression, early stopping, and RAG routers for model selection.
<argument name="citation_id">34</argument>


#### Required Cost Analysis
Assumptions for estimates:
- 1,000 daily queries (30,000/month).
- Average query: 1,000 input tokens (query + retrieved chunks), 200 output tokens.
- Document base: 10,000 pages, embedded once (batch cost).
- Embeddings: ~1M tokens total for ingestion.
- Infra: AWS EC2 GPU instance (e.g., g5.xlarge at ~$1.00/hour post-2025 reductions, running 24/7 ~$720/month).
<argument name="citation_id">12</argument>

<argument name="citation_id">13</argument>

- Scaled from benchmarks: RAG query cost ~$0.01-0.05 with GPT-4.
<argument name="citation_id">15</argument>

<argument name="citation_id">16</argument>


| Scenario | Components | Monthly Cost Breakdown | Total Estimated Monthly Cost |
|----------|------------|------------------------|------------------------------|
| **Premium Setup** (GPT-4 + Pinecone + Cloud Hosting) | - LLM: GPT-4.1 ($2/1M input, $8/1M output)<br>- Vector DB: Pinecone Standard ($50 min + usage)<br>- Hosting: AWS GPU for embeddings/processing<br>- Processing: Unstructured API (~$0.01/page) | - LLM: (30M input tokens: $60) + (6M output: $48) = $108<br>- Vector DB: $50 + $20 (queries/storage)<br>- Hosting: $720<br>- Processing: $100 (initial + updates) | ~$1,000 (High due to API calls; scales to $16k+ for heavier use)
<argument name="citation_id">20</argument>
 |
| **Optimized Setup** (Local Llama + Chroma + Self-Hosted Infra) | - LLM: Llama 3 via Ollama (free)<br>- Vector DB: Chroma (free)<br>- Hosting: On-prem server ($5k upfront amortized over 12 months ~$417/month + $100 electricity)<br>- Processing: Local unstructured (free) | - LLM: $0<br>- Vector DB: $0<br>- Hosting: $517<br>- Processing: $0 | ~$520 (Upfront GPU ~$2k-10k, but no per-query costs)
<argument name="citation_id">8</argument>

<argument name="citation_id">3</argument>
 |
| **Hybrid Approach** (Mix Local/Cloud) | - LLM: Tiered (80% local Llama, 20% GPT-4.1 mini)<br>- Vector DB: Chroma self-hosted<br>- Hosting: AWS for peak loads (~$300/month partial usage)<br>- Processing: Batch local, API for complex docs | - LLM: Local $0 + Cloud ($0.40/1M input * 6M: $2.40) + output $9.60 = $12<br>- Vector DB: $0<br>- Hosting: $300<br>- Processing: $50 | ~$360 (Balances cost and performance)
<argument name="citation_id">1</argument>

<argument name="citation_id">32</argument>
 |

#### Performance Trade-Offs Analysis
- **Premium**: Highest accuracy (GPT-4 excels at complex tables/regulations), but latency from API calls and high costs. No privacy issues if compliant.
- **Optimized**: Lower latency (local inference), zero API costs, but potentially reduced accuracy (Llama 3 ~80-90% of GPT-4 on benchmarks).
<argument name="citation_id">6</argument>
 Requires GPU hardware; fine-tuning needed for banking specifics.
- **Hybrid**: Good balance—use local for 80% queries (simple lookups), cloud for edge cases (e.g., ambiguous compliance). Trade-off: Added complexity in routing logic, but 50-80% cost savings.
<argument name="citation_id">32</argument>


#### Recommendations for Different Budget Scenarios
- **Low Budget (<$500/month)**: Go optimized with local Llama + Chroma on a single GPU server. Start with free HuggingFace embeddings.
- **Medium Budget ($500-1,000/month)**: Hybrid—local for base, GPT-4.1 mini for premium queries. Use batching and caching to cap costs.
- **High Budget (>$1,000/month)**: Premium for max accuracy, with Pinecone for scalable storage. Invest in fine-tuning for banking domain.

#### ROI Calculations for Banking Use Case
- **Assumptions**: Bank handles 1,000 queries/day from staff/customers. Manual handling costs $10/query (analyst time). RAG accuracy: 90% automated resolutions.
- **Savings**: 900 automated queries/day * $10 = $9,000/day * 30 = $270,000/month in labor savings.
- **Costs**: Premium ~$1,000/month; Optimized ~$520/month.
- **ROI**: Premium: ($270,000 - $1,000)/$1,000 = 269x monthly. Optimized: 519x. Break-even in <1 month; long-term, reduces compliance errors (potential $100k+ fines avoided).
<argument name="citation_id">22</argument>
 Hybrid maximizes ROI for scaling banks.