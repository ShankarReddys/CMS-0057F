# AI School of India — Video 1 Bonus Project

## Project
Build a ChatGPT-like chatbot using Python, Streamlit and Ollama.

This version does not require OpenAI API credits.

## What is Ollama?
Ollama allows you to run open-source LLMs locally on your own laptop.

Examples:
- Llama
- Mistral
- Gemma
- Qwen

## Architecture

User
↓
Streamlit UI
↓
Ollama Local Server
↓
Local LLM
↓
Response

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

