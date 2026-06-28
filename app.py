import os
import re
import requests
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Global Config & Helper Functions
# -----------------------------
PDF_DIR = "./pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

def check_ollama_running():
    try:
        response = requests.get("http://localhost:11434", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def tokenize(text):
    if not text:
        return []
    import string
    # Remove punctuation, convert to lowercase
    clean_text = "".join(char if char not in string.punctuation else " " for char in text)
    return [word.lower() for word in clean_text.split() if len(word) > 1]

def get_bm25_top_chunks(query, chunks, top_k=3):
    import math
    from collections import Counter
    query_tokens = tokenize(query)
    if not query_tokens or not chunks:
        return []
        
    doc_count = len(chunks)
    df = Counter()
    for chunk in chunks:
        unique_tokens = set(tokenize(chunk["text"]))
        for token in unique_tokens:
            df[token] += 1
            
    doc_lens = [len(tokenize(chunk["text"])) for chunk in chunks]
    avg_dl = sum(doc_lens) / doc_count if doc_count > 0 else 0
    
    k1 = 1.5
    b = 0.75
    
    scores = []
    for chunk in chunks:
        tokens = tokenize(chunk["text"])
        tf = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        
        for token in query_tokens:
            if tf[token] == 0:
                continue
            df_token = df[token]
            idf = math.log((doc_count - df_token + 0.5) / (df_token + 0.5) + 1.0)
            tf_sat = (tf[token] * (k1 + 1)) / (tf[token] + k1 * (1 - b + b * (doc_len / avg_dl if avg_dl > 0 else 1)))
            score += idf * tf_sat
            
        scores.append((score, chunk))
        
    scores = [(s, c) for s, c in scores if s > 0]
    scores.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scores[:top_k]]

def rebuild_pdf_index():
    pdf_chunks = []
    if os.path.exists(PDF_DIR):
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
        for filename in pdf_files:
            try:
                from pypdf import PdfReader
                reader = PdfReader(os.path.join(PDF_DIR, filename))
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pdf_chunks.append({
                            "filename": filename,
                            "page": i + 1,
                            "text": text.strip()
                        })
            except Exception as e:
                pass
    st.session_state.pdf_chunks = pdf_chunks

def render_obsidian_graph():
    chunks = st.session_state.get("pdf_chunks", [])
    if not chunks:
        st.warning("No documents indexed. Please index some PDFs first.")
        return

    # Combine text by filename
    doc_texts = {}
    for c in chunks:
        fname = c["filename"]
        doc_texts[fname] = doc_texts.get(fname, "") + " " + c["text"]

    if not doc_texts:
        return

    filenames = list(doc_texts.keys())
    texts = list(doc_texts.values())

    # Extract topics using TF-IDF
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        st.error(f"Error extracting topics: {e}")
        return

    nodes = []
    edges = []
    
    # 1. Create Document Nodes
    for fname in filenames:
        nodes.append(Node(id=fname, 
                          label=fname, 
                          size=25, 
                          color="#3b82f6", # Blue
                          shape="dot"))

    # 2. Find significant topics for each doc and create edges/topic nodes
    topic_nodes_added = set()
    dense_matrix = tfidf_matrix.todense()
    
    for i, fname in enumerate(filenames):
        doc_vector = dense_matrix[i].tolist()[0]
        # Get top 3 indices for this document
        top_indices = sorted(range(len(doc_vector)), key=lambda i: doc_vector[i], reverse=True)[:3]
        
        for idx in top_indices:
            if doc_vector[idx] > 0:
                topic = feature_names[idx]
                if topic not in topic_nodes_added:
                    nodes.append(Node(id=topic, 
                                      label=topic, 
                                      size=15, 
                                      color="#10b981", # Green
                                      shape="hexagon"))
                    topic_nodes_added.add(topic)
                
                edges.append(Edge(source=fname, 
                                  target=topic, 
                                  color="#cbd5e1",
                                  physics=True))

    config = Config(width=1000,
                    height=600,
                    directed=False, 
                    physics=True, 
                    hierarchical=False,
                    nodeHighlightBehavior=True,
                    highlightColor="#f59e0b",
                    collapsible=False)

    agraph(nodes=nodes, edges=edges, config=config)


def render_citations(text, sources):
    if not sources:
        return text
    for idx, src in enumerate(sources):
        marker = f"[{idx + 1}]"
        html_pill = f'<span class="citation-pill" title="{src["filename"]} (Page {src["page"]})">{idx + 1}</span>'
        text = text.replace(marker, html_pill)
    return text

def extract_followups(text):
    lines = text.split('\n')
    cleaned_lines = []
    followups = []
    for line in lines:
        match = re.search(r'\[FOLLOWUP\]\s*(?:[:\-]\s*)?(.*)', line, re.IGNORECASE)
        if match:
            clean_q = match.group(1).strip()
            # Strip markdown formatting like bold/italics and leading numbers
            clean_q = re.sub(r'^[\*\-\d\.\s]+', '', clean_q).strip('*_\"\' ')
            if clean_q:
                followups.append(clean_q)
        else:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines), followups


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Assistant",
    page_icon="✦",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
/* =============================================
   Clean Light UI — Card-based Design System
   Inspired by modern AI tool onboarding UIs
   ============================================= */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], [class*="st-key"], .stMarkdown, p, div, span,
button, input, textarea, select, label {
    font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
}

/* Strip Streamlit chrome */
header { visibility: hidden !important; height: 0 !important; }
footer { visibility: hidden !important; }
[data-testid="stHeader"] { background-color: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Main canvas ── */
.stApp {
    background-color: #f7f7f8 !important;
}
.block-container {
    max-width: 52rem !important;
    padding-top: 1rem !important;
    padding-bottom: 7rem !important;
}

/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e8e8e8 !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.03) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1.25rem !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #6b7280 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #6b7280 !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] hr {
    border-color: #e8e8e8 !important;
    margin: 0.75rem 0 !important;
}

/* Sidebar buttons – clean flat rows */
[data-testid="stSidebar"] .stButton button {
    background-color: transparent !important;
    color: #374151 !important;
    border: none !important;
    padding: 10px 14px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    min-height: auto !important;
    box-shadow: none !important;
    border-radius: 10px !important;
    margin-bottom: 2px !important;
    width: 100% !important;
    transform: none !important;
    transition: background-color 0.15s ease !important;
    line-height: 1.4 !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #f3f4f6 !important;
    color: #111827 !important;
    transform: none !important;
}
/* Active chat row */
[data-testid="stSidebar"] .stButton button[kind="primary"],
[data-testid="stSidebar"] .stButton button[data-testid*="primary"] {
    background-color: #f3f4f6 !important;
    color: #111827 !important;
    font-weight: 600 !important;
}
/* New-Chat button */
[data-testid="stSidebar"] .stButton button[class*="new_chat_btn"] {
    border: 1px solid #d1d5db !important;
    background-color: #ffffff !important;
    margin-bottom: 1rem !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    color: #374151 !important;
}
[data-testid="stSidebar"] .stButton button[class*="new_chat_btn"]:hover {
    background-color: #f9fafb !important;
    border-color: #9ca3af !important;
}
/* Settings / nav buttons */
[data-testid="stSidebar"] .stButton button[class*="settings_nav_btn"],
[data-testid="stSidebar"] .stButton button[class*="chat_nav_btn"] {
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    color: #6b7280 !important;
    font-size: 13px !important;
    margin-top: 0.25rem !important;
    background-color: #ffffff !important;
}
[data-testid="stSidebar"] .stButton button[class*="settings_nav_btn"]:hover,
[data-testid="stSidebar"] .stButton button[class*="chat_nav_btn"]:hover {
    background-color: #f3f4f6 !important;
    color: #111827 !important;
}

/* ── Chat Messages ───────────────────────── */
.stChatMessage, [data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    border-bottom: none !important;
    padding: 1.25rem 0 !important;
    margin: 0 !important;
    gap: 1rem !important;
}
.stChatMessage:has(div[class*="st-key-user_msg"]),
[data-testid="stChatMessage"]:has(div[class*="st-key-user_msg"]) {
    flex-direction: row !important;
}

/* User message – subtle gray bubble */
div[class*="st-key-user_msg"] {
    background-color: #e9eaec !important;
    color: #1a1a1a !important;
    padding: 12px 18px !important;
    border-radius: 20px !important;
    max-width: 85% !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    border: none !important;
    text-align: left !important;
    line-height: 1.6 !important;
}

/* Assistant message – clean text */
div[class*="st-key-assistant_msg"] {
    background-color: transparent !important;
    color: #374151 !important;
    padding: 4px 0px !important;
    max-width: 100% !important;
    margin-left: 0 !important;
    line-height: 1.7 !important;
}

/* Avatars */
[data-testid="stChatMessage"] [data-testid="stAvatar"] {
    background-color: #f3f4f6 !important;
    border-radius: 50% !important;
    border: 1px solid #e5e7eb !important;
    width: 32px !important;
    height: 32px !important;
}

/* ── Chat Input ──────────────────────────── */
[data-testid="stChatInputContainer"] {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 26px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
    padding: 4px 12px !important;
    max-width: 52rem !important;
    margin: 0 auto !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: #9ca3af !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    color: #1a1a1a !important;
    background-color: transparent !important;
    border: none !important;
    font-size: 15px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #9ca3af !important;
}
/* Send button */
[data-testid="stChatInputContainer"] button {
    background-color: #1a1a1a !important;
    border: none !important;
    border-radius: 50% !important;
    color: #fff !important;
    min-height: auto !important;
    box-shadow: none !important;
    padding: 6px !important;
}
[data-testid="stChatInputContainer"] button:hover {
    background-color: #374151 !important;
    transform: none !important;
}

/* ── Suggestion Cards ────────────────────── */
.stButton button {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    color: #374151 !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    min-height: 64px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    line-height: 1.5 !important;
}
.stButton button:hover {
    background-color: #f9fafb !important;
    border-color: #1a1a1a !important;
    color: #111827 !important;
    transform: none !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}

/* ── Settings Page ───────────────────────── */
.stSelectbox label, .stSlider label, .stTextArea label, .stCheckbox label {
    color: #374151 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
.stSelectbox [data-testid="stSelectbox"] > div {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    color: #1a1a1a !important;
}
.stTextArea textarea {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    color: #1a1a1a !important;
    font-size: 13px !important;
}
.stCheckbox label span {
    color: #374151 !important;
}
.stSlider [data-testid="stSlider"] {
    color: #1a1a1a !important;
}

/* Expander / Sources */
.streamlit-expanderHeader {
    color: #6b7280 !important;
    font-size: 13px !important;
}
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    background-color: #ffffff !important;
}

/* ── Citation pills ──────────────────────── */
.citation-pill {
    background-color: #f0fdf4 !important;
    color: #16a34a !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    padding: 2px 7px !important;
    margin: 0 2px !important;
    cursor: pointer !important;
    font-weight: 600 !important;
    display: inline-block !important;
    vertical-align: middle !important;
    border: 1px solid #bbf7d0 !important;
    transition: all 0.15s ease !important;
}
.citation-pill:hover {
    background-color: #dcfce7 !important;
    color: #15803d !important;
}

/* ── Scrollbar ───────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9ca3af; }

/* ── Misc ─────────────────────────────────── */
.stCaption, [data-testid="stCaption"] {
    color: #9ca3af !important;
}
a { color: #2563eb !important; }
a:hover { color: #1d4ed8 !important; }

/* ── Model cards (settings page) ─────────── */
.model-card {
    background: #ffffff;
    border: 1.5px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px;
    cursor: pointer;
    transition: all 0.15s ease;
    position: relative;
}
.model-card:hover {
    border-color: #1a1a1a;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.model-card.selected {
    border-color: #1a1a1a;
    box-shadow: 0 0 0 1px #1a1a1a;
}
.model-card .check {
    position: absolute;
    top: 12px;
    right: 12px;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 1.5px solid #d1d5db;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    color: transparent;
}
.model-card.selected .check {
    background-color: #1a1a1a;
    border-color: #1a1a1a;
    color: #ffffff;
}
.model-card .model-name {
    font-size: 15px;
    font-weight: 600;
    color: #1a1a1a;
    margin-top: 8px;
}
.model-card .model-desc {
    font-size: 13px;
    color: #6b7280;
    margin-top: 4px;
    line-height: 1.4;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Ollama Settings
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"

# -----------------------------
# Session State / Memory
# -----------------------------
PDF_DIR = "./pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "pdf_cache" not in st.session_state:
    st.session_state.pdf_cache = {}

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "active_page" not in st.session_state:
    st.session_state.active_page = "chat"

if "model" not in st.session_state:
    st.session_state.model = "gemma2:2b"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """You are CMS-0057F Assistant, an expert guide for a U.S. health plan on the CMS
Interoperability and Prior Authorization Final Rule (CMS-0057-F). You serve two
audiences: Health Plan IT/technical staff and business/operations staff. Adapt
depth and vocabulary to whichever the user appears to be.

KNOWLEDGE SOURCE
- Your answers must be grounded in the retrieved excerpts from the plan's indexed
  PDF library on CMS-0057F (rule text, implementation guides, internal analyses,
  vendor materials, and compliance plans).
- Give a priority for the information to document vectors that are generated from /pdfs folder over the LLM context. that is first 

- Treat the retrieved excerpts as the source of truth. When they conflict with your
  own prior knowledge, defer to the documents and note the discrepancy.
- Do NOT rely on memorized dates, thresholds, timeframes, or metrics. These are the
  details most likely to be wrong from memory — always pull them from the retrieved
  context. If a specific date or number is not in the retrieved excerpts, say so
  rather than guessing.

GROUNDING & CITATION RULES
- Every substantive claim must trace to the retrieved excerpts. After each answer,
  cite sources as [document name, p.X]. If multiple documents support a point, cite
  each.
- If the retrieved context does not contain the answer, state clearly: "I don't have
  that in the indexed documents," then suggest where the user might look or what to
  add to the library. Never fabricate citations, page numbers, or requirements.
- Distinguish clearly between (a) what the rule requires, (b) what an implementation
  guide recommends, and (c) what is internal plan interpretation or vendor opinion.

AUDIENCE ADAPTATION
- For IT/technical users: be precise about FHIR APIs (Patient Access, Provider
  Access, Payer-to-Payer, Prior Authorization), HL7 Da Vinci IGs (CRD, DTR, PAS,
  PDex), data elements, authentication, and bulk/operational patterns. Use exact
  terminology from the documents.
- For business/operations users: lead with plain-language meaning, operational
  impact, deadlines, reporting obligations, and member/provider effects. Define
  acronyms on first use.
- When the audience is unclear, give a short plain-language answer first, then offer
  to go deeper on the technical side.

RESPONSE STYLE
- Be concise and structured. Default to short answer first, then supporting detail.
- Use headers, bullets, or small tables when they aid clarity (e.g., requirement →
  deadline → impacted payer types → status).
- Quote rule text sparingly and only when exact wording matters; otherwise
  paraphrase and cite.
- When asked "what do we need to do," frame as actionable items, and flag any
  dependency on internal decisions not found in the documents.

SCOPE & GUARDRAILS
- Stay within CMS-0057F and directly related interoperability/prior-authorization
  topics. If asked something out of scope, say so and redirect.
- STRICT DATA SOURCE GUARDRAIL: You must ONLY answer questions based on the retrieved 
  excerpts from the /pdfs directory. Do not use your general training data to answer 
  questions. If the retrieved context does not contain the answer, you must refuse to 
  answer and state that the information is not available in the provided documents.
- You provide informational support, not legal or compliance sign-off. For binding
  determinations, advise the user to confirm with compliance/legal and the official
  rule text in the Federal Register.
- Do not expose any PHI; if a user includes member-identifiable data, do not repeat
  it back and remind them not to enter PHI.
- If a question depends on which payer type applies (Medicare Advantage, Medicaid,
  CHIP, QHP issuers), ask or state your assumption, since requirements and dates
  differ by program.

When you lack enough retrieved context to answer well, ask one focused clarifying
question rather than producing a vague answer."""

# Migrate legacy messages if they exist
if "messages" in st.session_state and st.session_state.messages:
    if not st.session_state.chats:
        import uuid
        first_msg = st.session_state.messages[0]["content"]
        title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
        migrated_id = str(uuid.uuid4())
        st.session_state.chats[migrated_id] = {
            "title": title,
            "messages": st.session_state.messages
        }
        st.session_state.current_chat_id = migrated_id

# Ensure there's at least one active chat session
if not st.session_state.chats:
    import uuid
    default_id = str(uuid.uuid4())
    st.session_state.chats[default_id] = {
        "title": "New Chat",
        "messages": []
    }
    st.session_state.current_chat_id = default_id

# Reference the active chat's message list
current_chat = st.session_state.chats[st.session_state.current_chat_id]
st.session_state.messages = current_chat["messages"]

with st.sidebar:
    # App logo / brand
    st.markdown('<div style="padding: 0 0 1rem 2px;"><span style="font-size: 22px;">✦</span> <span style="font-size: 16px; font-weight: 600; color: #1a1a1a; vertical-align: middle;">AI Assistant</span></div>', unsafe_allow_html=True)
    
    if st.button("＋  New chat", key="new_chat_btn", use_container_width=True):
        import uuid
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {
            "title": "New Chat",
            "messages": []
        }
        st.session_state.current_chat_id = new_id
        st.rerun()
    
    st.markdown("### Recent")
    for chat_id, chat_data in list(st.session_state.chats.items())[::-1]:
        title = chat_data["title"]
        is_active = (chat_id == st.session_state.current_chat_id)
        if is_active:
            st.button(f"{title}", key=f"active_{chat_id}", use_container_width=True, type="primary")
        else:
            if st.button(f"{title}", key=f"inactive_{chat_id}", use_container_width=True, type="secondary"):
                st.session_state.current_chat_id = chat_id
                st.rerun()

    st.markdown("---")
    st.markdown("---")
    
    if st.session_state.active_page != "chat":
        if st.button("← Back to Chat", use_container_width=True, key="chat_nav_btn"):
            st.session_state.active_page = "chat"
            st.rerun()


    if st.session_state.active_page != "settings":
        if st.button("⚙  Settings", use_container_width=True, key="settings_nav_btn"):
            st.session_state.active_page = "settings"
            st.rerun()




# -----------------------------
# Ollama Health Check
# -----------------------------
if not check_ollama_running():
    st.error("Ollama is not running. Please open terminal and run: ollama serve")
    st.info("Then pull a model using: ollama pull llama3.2")
    st.stop()

# -----------------------------
# Settings Page View
# -----------------------------
if st.session_state.active_page == "settings":
    st.markdown('<h1 style="font-size: 24px; font-weight: 700; color: #1a1a1a; margin-bottom: 4px;">⚙️ Settings</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6b7280; font-size: 14px; margin-bottom: 2rem;">Configure your model, system instructions, and document index.</p>', unsafe_allow_html=True)
    
    # ── Select Model (card-based UI) ──
    st.markdown('<h3 style="font-size: 15px; font-weight: 600; color: #1a1a1a; margin-bottom: 0.75rem;">Select Model</h3>', unsafe_allow_html=True)
    
    model_options = [
        {"id": "gemma2:2b",   "name": "Gemma 2 (2B)",   "desc": "Google's lightweight model optimized for fast local inference."},
        {"id": "llama3.2",    "name": "Llama 3.2",       "desc": "Meta's latest compact model for efficient general tasks."},
        {"id": "llama3.1",    "name": "Llama 3.1",       "desc": "Meta's performant model with strong reasoning abilities."},
        {"id": "llama3",      "name": "Llama 3",         "desc": "Meta's foundational open model for broad capabilities."},
        {"id": "mistral",     "name": "Mistral",         "desc": "Mistral AI's efficient model excels at instruction following."},
        {"id": "gemma2",      "name": "Gemma 2",         "desc": "Google's mid-size model balancing quality and speed."},
        {"id": "qwen2.5",     "name": "Qwen 2.5",        "desc": "Alibaba's multilingual model with strong coding skills."},
        {"id": "qwen2.5:3b",  "name": "Qwen 2.5 (3B)",   "desc": "Compact variant of Qwen 2.5 for resource-limited setups."},
    ]
    
    # Render model cards as HTML
    card_html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 1.5rem;">'
    for m in model_options:
        is_selected = (m["id"] == st.session_state.model)
        sel_class = "selected" if is_selected else ""
        check_icon = "✓" if is_selected else ""
        card_html += f'''
        <div class="model-card {sel_class}" onclick="void(0)">
            <div class="check">{check_icon}</div>
            <div class="model-name">{m["name"]}</div>
            <div class="model-desc">{m["desc"]}</div>
        </div>'''
    card_html += '</div>'
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Functional selector (syncs with cards visually)
    models_list = [m["id"] for m in model_options]
    current_idx = models_list.index(st.session_state.model) if st.session_state.model in models_list else 0
    selected_model = st.selectbox(
        "Active model",
        models_list,
        index=current_idx,
        format_func=lambda x: next((m["name"] for m in model_options if m["id"] == x), x)
    )
    st.session_state.model = selected_model
    
    st.markdown("---")
    
    # Temperature
    st.markdown('<h3 style="font-size: 15px; font-weight: 600; color: #1a1a1a; margin-bottom: 0.5rem;">Temperature</h3>', unsafe_allow_html=True)
    selected_temp = st.slider(
        "Creativity level",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="0 = focused and precise, 1 = creative and varied",
        label_visibility="collapsed"
    )
    st.session_state.temperature = selected_temp
    st.caption(f"Current: {selected_temp} — {'Precise' if selected_temp < 0.4 else 'Balanced' if selected_temp < 0.7 else 'Creative'}")
    
    st.markdown("---")
    
    # System Prompt
    st.markdown('<h3 style="font-size: 15px; font-weight: 600; color: #1a1a1a; margin-bottom: 0.5rem;">System Prompt</h3>', unsafe_allow_html=True)
    selected_prompt = st.text_area(
        "System instructions",
        value=st.session_state.system_prompt,
        height=200,
        label_visibility="collapsed"
    )
    st.session_state.system_prompt = selected_prompt
    
    st.markdown("---")
    
    # PDF RAG
    st.markdown('<h3 style="font-size: 15px; font-weight: 600; color: #1a1a1a; margin-bottom: 0.5rem;">📄 Document Library</h3>', unsafe_allow_html=True)
    
    pdf_rag = st.checkbox(
        "Enable PDF RAG (Search All Documents)", 
        value=st.session_state.get("pdf_rag_enabled", True)
    )
    st.session_state.pdf_rag_enabled = pdf_rag
    
    if st.button("🔄  Re-Index Documents", use_container_width=True):
        with st.spinner("Re-indexing documents..."):
            rebuild_pdf_index()
        st.success("Document index rebuilt!")
        st.rerun()
        
    total_pages = len(st.session_state.get("pdf_chunks", []))
    total_docs = len(set(c["filename"] for c in st.session_state.pdf_chunks)) if total_pages > 0 else 0
    st.caption(f"Index: {total_pages} pages from {total_docs} PDFs")
    
    st.markdown("---")
    
    # Danger Zone
    st.markdown('<h3 style="font-size: 15px; font-weight: 600; color: #dc2626; margin-bottom: 0.5rem;">Danger Zone</h3>', unsafe_allow_html=True)
    if st.button("🗑️  Delete Current Chat", use_container_width=True, type="primary"):
        del st.session_state.chats[st.session_state.current_chat_id]
        if not st.session_state.chats:
            import uuid
            default_id = str(uuid.uuid4())
            st.session_state.chats[default_id] = {
                "title": "New Chat",
                "messages": []
            }
            st.session_state.current_chat_id = default_id
        else:
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[-1]
        st.session_state.active_page = "chat"
        st.rerun()
        
    st.stop()

# -----------------------------
# Graph Page View
# -----------------------------
elif st.session_state.active_page == "graph":
    st.markdown('<h1 style="font-size: 24px; font-weight: 700; color: #1a1a1a; margin-bottom: 4px;">📈 Obsidian Graph View</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 14px; color: #6b7280; margin-bottom: 2rem;">Visualizing connections between chunks in your Document Library.</p>', unsafe_allow_html=True)
    
    if not st.session_state.get("pdf_chunks"):
        st.info("No documents are currently indexed. Go to Settings to rebuild the index or upload PDFs.")
    else:
        with st.spinner("Generating document graph (this might take a few seconds)..."):
            nodes, edges = create_document_graph(st.session_state.pdf_chunks)
            
            if not nodes:
                st.warning("Not enough documents or content to generate meaningful relationships.")
            else:
                config = Config(
                    width=900,
                    height=600,
                    directed=False, 
                    physics=True, 
                    hierarchical=False,
                    nodeHighlightBehavior=True,
                    highlightColor="#10b981", # Emerald green
                    collapsible=False,
                    node={'labelProperty': 'label'},
                    link={'labelProperty': 'label', 'renderLabel': True}
                )
                
                # Render the agraph component
                return_value = agraph(nodes=nodes, edges=edges, config=config)
                
                st.caption(f"Graph generated with {len(nodes)} nodes and {len(edges)} edges. Nodes represent document chunks. Edges represent TF-IDF similarity ≥ 0.25.")
    
    # Stop execution so chat doesn't render below
    st.stop()


# -----------------------------
# Chat Interface (Main)
# -----------------------------
elif st.session_state.active_page == "chat":
    # Minimal model badge at top of chat
    st.markdown(f'<div style="text-align: center; padding: 0.25rem 0 0.75rem 0;"><span style="font-size: 14px; font-weight: 600; color: #374151;">{st.session_state.model}</span> <span style="font-size: 12px; color: #9ca3af;">· {st.session_state.temperature} temp</span></div>', unsafe_allow_html=True)

    # -----------------------------
    # Show Chat History
    # -----------------------------
    for i, message in enumerate(st.session_state.messages):
        avatar = "👤" if message["role"] == "user" else "✨"
        key = f"user_msg_{i}" if message["role"] == "user" else f"assistant_msg_{i}"
        with st.chat_message(message["role"], avatar=avatar):
            with st.container(key=key):
                display_content = message["content"]
                followups = []
                if message["role"] == "assistant":
                    display_content, followups = extract_followups(message["content"])
                    
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    rendered_text = render_citations(display_content, message["sources"])
                    st.markdown(rendered_text, unsafe_allow_html=True)
                else:
                    st.markdown(display_content)
                    
                if followups:
                    st.write("")
                    num_followups = len(followups)
                    spacer_weight = max(1, 4 - num_followups)
                    cols = st.columns([spacer_weight] + [1.5] * num_followups)
                    
                    for idx, q in enumerate(followups):
                        if cols[idx+1].button(f"{q} ↗", key=f"hist_followup_{i}_{idx}", use_container_width=True, type="tertiary"):
                            st.session_state.messages.append({"role": "user", "content": q})
                            st.rerun()
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander("🔍 Reference Sources Used"):
                        for src in message["sources"]:
                            st.markdown(f"- **{src['filename']}** (Page {src['page']})")
                            st.caption(f"\"{src['text'][:250]}...\"")

    # -----------------------------
    # Chat Input
    # -----------------------------
    user_prompt = st.chat_input("Message AI Assistant...")

    if user_prompt:
        if len(st.session_state.messages) == 0:
            st.session_state.chats[st.session_state.current_chat_id]["title"] = user_prompt[:30] + "..." if len(user_prompt) > 30 else user_prompt
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.rerun()

    # -----------------------------
    # Landing Page / Suggestion Cards
    # -----------------------------
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div style="text-align: center; margin-top: 6rem; margin-bottom: 2.5rem;">
            <div style="font-size: 32px; margin-bottom: 12px;">✦</div>
            <div style="font-size: 24px; font-weight: 700; color: #1a1a1a; margin-bottom: 6px;">What can I help with?</div>
            <div style="font-size: 14px; color: #9ca3af;">Ask a question about your documents or anything else.</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋  Explain CMS-0057F PA\nInteroperability requirements", use_container_width=True):
                prompt = "What is CMS-0057F PA Interoperability?"
                st.session_state.chats[st.session_state.current_chat_id]["title"] = prompt[:30] + "..." if len(prompt) > 30 else prompt
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
            if st.button("✉️  What is Consent management requirements for CMS-0057F?", use_container_width=True):
                prompt = "What is Consent management requirements for CMS-0057F?"
                st.session_state.chats[st.session_state.current_chat_id]["title"] = prompt[:30] + "..." if len(prompt) > 30 else prompt
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
        with col2:
            if st.button("💡  What are specifications for Patient API and Provider API?", use_container_width=True):
                prompt = "What are specifications for Patient API and Provider API?"
                st.session_state.chats[st.session_state.current_chat_id]["title"] = prompt[:30] + "..." if len(prompt) > 30 else prompt
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
            if st.button("🐍  Write a Python function\nto safely parse JSON", use_container_width=True):
                prompt = "Write a Python function to parse JSON safely"
                st.session_state.chats[st.session_state.current_chat_id]["title"] = prompt[:30] + "..." if len(prompt) > 30 else prompt
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

    # -----------------------------
    # Generate Assistant Response
    # -----------------------------
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        sys_prompt = st.session_state.system_prompt + "\n\nCRITICAL INSTRUCTION: At the very end of your response, you MUST provide 1 to 3 suggested follow-up questions that the user could ask next. Format each question on a new line starting exactly with '[FOLLOWUP] '."
        messages_for_ollama = [{"role": "system", "content": sys_prompt}]
        retrieved_sources = []
        
        # Retrieve & Inject Multi-Document PDF context if RAG is enabled
        if st.session_state.get("pdf_rag_enabled", True) and st.session_state.get("pdf_chunks"):
            user_query = st.session_state.messages[-1]["content"]
            retrieved_sources = get_bm25_top_chunks(user_query, st.session_state.pdf_chunks, top_k=3)
            
            if retrieved_sources:
                context_blocks = []
                for j, src in enumerate(retrieved_sources):
                    context_blocks.append(
                        f"--- START CONTEXT {j+1} ---\n"
                        f"Source document: '{src['filename']}', Page: {src['page']}\n"
                        f"Content:\n{src['text']}\n"
                        f"--- END CONTEXT {j+1} ---\n\n"
                    )
                
                rag_system_message = (
                    f"You have access to the following relevant document contexts retrieved from the PDF Document Library:\n\n"
                    f"{''.join(context_blocks)}"
                    f"INSTRUCTIONS:\n"
                    f"1. You MUST prioritize the facts inside the retrieved contexts above to answer the user's question.\n"
                    f"2. You MUST cite your sources inline using numerical bracket markers corresponding to the context number, e.g. [1], [2], [3]. Put them inline directly after the sentence referencing those facts (e.g., 'The passcode is XYZ [1].'). Do NOT cite them at the very end of the response; place the brackets directly inside the body paragraphs.\n"
                    f"3. If the answer is not mentioned in the contexts, state clearly that it is not found in the documents, and then use your general knowledge, explicitly mentioning that you are using general knowledge.\n"
                    f"4. Keep your explanations accurate and directly reference the citations."
                )
                messages_for_ollama.append({"role": "system", "content": rag_system_message})

        messages_for_ollama.extend(st.session_state.messages)

        with st.chat_message("assistant", avatar="✨"):
            with st.container(key="assistant_msg_streaming"):
                response_placeholder = st.empty()
                full_response = ""

                try:
                    payload = {
                        "model": st.session_state.model,
                        "messages": messages_for_ollama,
                        "stream": True,
                        "options": {
                            "temperature": st.session_state.temperature
                        }
                    }

                    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as response:
                        response.raise_for_status()

                        for line in response.iter_lines():
                            if line:
                                data = line.decode("utf-8")
                                import json
                                chunk = json.loads(data)

                                if "message" in chunk and "content" in chunk["message"]:
                                    content = chunk["message"]["content"]
                                    full_response += content
                                    clean_temp, _ = extract_followups(full_response)
                                    rendered_temp = render_citations(clean_temp, retrieved_sources)
                                    response_placeholder.markdown(rendered_temp + "▌", unsafe_allow_html=True)

                                if chunk.get("done", False):
                                    break

                    clean_final, final_followups = extract_followups(full_response)
                    response_placeholder.markdown(render_citations(clean_final, retrieved_sources), unsafe_allow_html=True)

                except requests.exceptions.HTTPError as e:
                    full_response = (
                        f"HTTP Error: {str(e)}\n\n"
                        f"Most likely the model '{st.session_state.model}' is not downloaded.\n\n"
                        f"Run this command in terminal:\n\n"
                        f"ollama pull {st.session_state.model}"
                    )
                    response_placeholder.error(full_response)

                except Exception as e:
                    full_response = f"Error: {str(e)}"
                    response_placeholder.error(full_response)

        assistant_msg = {"role": "assistant", "content": full_response}
        if retrieved_sources:
            assistant_msg["sources"] = retrieved_sources
        st.session_state.messages.append(assistant_msg)
        st.rerun()
