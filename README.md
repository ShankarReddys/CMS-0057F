# CMS-0057F Health Plan AI Assistant

## Project Overview
This project provides a local, privacy-preserving AI Assistant designed specifically for U.S. Health Plans to navigate and implement the **CMS Interoperability and Prior Authorization Final Rule (CMS-0057-F)**. 

By leveraging local LLMs (via Ollama) and a Retrieval-Augmented Generation (RAG) pipeline against your own indexed PDF library, this chatbot helps Health Plan IT/technical staff and business/operations staff quickly find grounded, cited answers regarding compliance dates, API specifications (Patient Access, Provider Access, Payer-to-Payer, DRLS), and prior authorization requirements without sending sensitive patient or proprietary data to the cloud.

This version runs entirely locally and does not require OpenAI API credits or external internet access for inference.

## What is Ollama?
Ollama allows you to run open-source LLMs locally on your own laptop.

Examples:
- Llama
- Mistral
- Gemma
- Qwen

## Architecture

This diagram illustrates the end-to-end architecture of your local AI Assistant, highlighting the Streamlit frontend, the lightweight BM25 Retrieval-Augmented Generation (RAG) pipeline, and the local Ollama LLM integration.

```mermaid
graph TD
    subgraph Frontend["🖥️ Frontend (Streamlit)"]
        UI[User Interface]
        State[Session State Management]
        Settings[Settings & Model Selection]
    end

    subgraph Data["📂 Local Storage"]
        PDFs[PDF Documents in /pdfs]
    end

    subgraph Indexing["⚙️ Offline Indexing Pipeline"]
        Parser[PDF Parser]
        Chunker[Text Chunker]
        BM25[BM25 Lexical Indexer]
        MemCache[(In-Memory Chunk Cache)]
    end

    subgraph Retrieval["🔍 RAG Retrieval Pipeline"]
        Tokenizer[Query Tokenizer]
        Searcher[BM25 Scoring Engine]
        ContextBuilder[Prompt Augmentation]
    end

    subgraph Backend["🧠 LLM Engine"]
        Ollama[Ollama Local Server]
        Models[(Local Models: Gemma, Llama, etc.)]
    end

    %% User interactions
    UI -->|1. User asks question| Tokenizer
    Settings -->|Configures| State
    
    %% Indexing Flow
    PDFs -->|Read| Parser
    Parser -->|Raw Text| Chunker
    Chunker -->|Text Chunks| BM25
    Chunker -->|Save Chunks| MemCache
    BM25 -->|Create TF-IDF Index| MemCache

    %% Retrieval Flow
    Tokenizer -->|Tokens| Searcher
    Searcher -->|Query Index| MemCache
    MemCache -->|Top-K Chunks| ContextBuilder
    
    %% Generation Flow
    UI -->|Chat History & Query| ContextBuilder
    ContextBuilder -->|Constructed Payload| Ollama
    Ollama <-->|Load/Run| Models
    Ollama -->|Stream Response| UI
    
    %% Styling
    classDef frontend fill:#f3f4f6,stroke:#d1d5db,stroke-width:2px,color:#111827
    classDef data fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,color:#111827
    classDef pipeline fill:#e0e7ff,stroke:#6366f1,stroke-width:2px,color:#111827
    classDef backend fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#111827
    
    class UI,State,Settings frontend
    class PDFs data
    class Parser,Chunker,BM25,MemCache pipeline
    class Tokenizer,Searcher,ContextBuilder pipeline
    class Ollama,Models backend
```

### Key Components

> [!TIP]
> This architecture is fully local and runs entirely on your machine without requiring external API calls or cloud dependencies.

1. **Frontend**: Built with Streamlit, handling the user interface, session state (chat history), and application settings.
2. **Indexing**: Instead of a vector database, this solution uses a lightweight **BM25 lexical search algorithm**. It tokenizes the text from the PDFs and scores them based on term frequency (TF-IDF), storing the chunks in memory.
3. **Retrieval**: When a query is made, it is tokenized and scored against the BM25 index to extract the Top-K most relevant document chunks.
4. **LLM Engine**: The augmented prompt (containing the system instructions, chat history, user query, and retrieved document context) is sent to a local **Ollama** server, which streams the generated response back to the UI.

## Step 1 — Install Ollama

Download and install Ollama from:

https://ollama.com

After installation, open terminal and check:

```bash
ollama --version
```

## Step 2 — Pull a model

Recommended beginner model:

```bash
ollama pull llama3.2
```

If your laptop has less RAM, try smaller models if available.

Other options:

```bash
ollama pull llama3
ollama pull mistral
ollama pull gemma2
ollama pull qwen2.5
```

## Step 3 — Run Ollama

Usually Ollama runs automatically after installation.

If not, run:

```bash
ollama serve
```

Keep this terminal open.

## Step 4 — Create virtual environment

```bash
python -m venv venv
```

## Step 5 — Activate virtual environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

## Step 6 — Install packages

```bash
pip install -r requirements.txt
```

## Step 7 — Run Streamlit app

```bash
streamlit run app.py
```

## Important
If you select a model in the sidebar, that model must be downloaded first.

Example:

If selected model is `mistral`, run:

```bash
ollama pull mistral
```

