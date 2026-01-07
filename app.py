import streamlit as st
from dotenv import load_dotenv
import os
from custom_llm import GithubGPT41LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from PyPDF2 import PdfReader

load_dotenv()

def load_pdfs_from_fileobjs(pdf_fileobjs):
    documents = []
    for fileobj in pdf_fileobjs:
        reader = PdfReader(fileobj)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(page_content=text, metadata={"page": i+1}))
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

def chroma_database(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma.from_documents(chunks, embeddings)

@st.cache_resource(show_spinner="Processing PDFs...")
def process_pdfs(uploaded_files):
    docs = load_pdfs_from_fileobjs(uploaded_files)
    chunks = split_text(docs)
    db = chroma_database(chunks)
    return db

# --- Custom CSS for ChatGPT-like UI ---
st.markdown("""
    <style>
    body {
        background-color: #1a1a1a;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
        margin: auto;
    }
    .stApp {
        background-color: #1a1a1a;
    }
    .chat-container {
        max-width: 700px;
        margin: 2rem auto 0 auto;
        padding: 2rem;
        background: #232634;
        border-radius: 16px;
        box-shadow: 0 2px 24px 0 rgba(0,0,0,0.2);
    }
    .chat-bubble {
        background: #232634;
        border-radius: 10px;
        padding: 1em;
        margin-bottom: 1em;
        color: #fff;
        font-size: 1.1em;
        line-height: 1.5;
        word-break: break-word;
    }
    .user-bubble {
        background: #FF4B4B;
        color: #fff;
        border-radius: 10px;
        padding: 1em;
        margin-bottom: 1em;
        font-size: 1.1em;
        line-height: 1.5;
        word-break: break-word;
    }
    .stTextInput>div>div>input {
        font-size: 1.2em;
        padding: 1em;
        border-radius: 8px;
        background: #232634;
        color: #fff;
        border: 1px solid #444;
    }
    .stButton>button {
        color: #FF4B4B;
        border: 1px solid #FF4B4B;
        border-radius: 8px;
        padding: 0.5em 1.5em;      /* Reduce horizontal padding */
        font-weight: bold;
        background: #232634;
        transition: 0.2s;
        font-size: 1.1em;
        /* width: 100%; */          /* Remove this line */
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 0;              /* Ensure no top margin */
        height: 48px;               /* Match input height (adjust as needed) */
    }
    .stButton>button:hover {
        background: #FF4B4B;
        color: white;
    }
    .uploaded-file {
        color: #fff;
        font-size: 1em;
        margin-bottom: 0.5em;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar for App Title and Chat History ---
with st.sidebar:
    st.markdown("<h2 style='color:#FF4B4B;margin-bottom:1em;'>AskMyPDF</h2>", unsafe_allow_html=True)
    # Chat History in Sidebar
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.markdown("#### 🕑 Chat History")
        for idx, (q, a) in enumerate(st.session_state.chat_history[::-1]):
            anchor = f"q{len(st.session_state.chat_history)-idx}"
            st.markdown(
                f"<a href='#{anchor}' style='color:#fff; text-decoration:none; font-size:0.95em; display:block; margin-bottom:0.5em;'>"
                f"Q{len(st.session_state.chat_history)-idx}: {q[:40]}{'...' if len(q)>40 else ''}"
                f"</a>",
                unsafe_allow_html=True
            )
    # Uploaded Files in Sidebar
    if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
        st.markdown("---")
        st.markdown("#### Uploaded Files")
        for file in st.session_state.uploaded_files:
            st.markdown(f"<div class='uploaded-file'>📄 {file.name}</div>", unsafe_allow_html=True)

# --- Main Area ---
st.markdown("<h1 style='color:#FF4B4B; text-align:center; margin-bottom: 0.5em;'>AskMyPDF</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#fff;text-align:center;'>What are you working on?</h3>", unsafe_allow_html=True)


# Main input area: question input at the top, then file uploader
def handle_question():
    question = st.session_state.user_input
    if question and (not st.session_state.chat_history or question != st.session_state.chat_history[-1][0]):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((question, result["answer"]))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

question = st.text_input(
    "",
    placeholder="Ask anything",
    key="user_input"
)

# --- PDF Upload and Processing ---
uploaded_files = st.file_uploader(
    "Upload PDF files below:",
    type=["pdf"],
    accept_multiple_files=True,
    key="main_uploader"
)
if uploaded_files:
    for file in uploaded_files:
        if not file.name.lower().endswith(".pdf"):
            st.error(f"❌ Unsupported file type: {file.name}")
            st.stop()

    st.session_state.uploaded_files = uploaded_files  # Save for sidebar display
    try:
        db = process_pdfs(uploaded_files)
    except Exception as e:
        st.error(f"❌ Failed to process PDF(s): {e}")
        st.stop()

    llm = GithubGPT41LLM(
        api_key=os.getenv("MARKETPLACE_API_KEY"),
        api_url=os.getenv("MARKETPLACE_API_URL"),
        model="openai/gpt-4.1",
        temperature=1.0,
        top_p=1.0
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Only trigger when question is not empty and is new
    if question and (not st.session_state.chat_history or question != st.session_state.chat_history[-1][0]):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
                st.session_state.chat_history.append((question, result["answer"]))
                st.markdown(f'<div class="chat-bubble"><b>Bot:</b> {result["answer"]}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ LLM Error: {e}")
                st.stop()
    # --- Clear PDFs and Chat Button ---
    col_clear, col_sum = st.columns([1, 1])
    with col_clear:
        if st.button("🗑️ Clear PDFs & Chat"):
            st.session_state.pop("uploaded_files", None)
            st.session_state.chat_history = []
            st.rerun()

    with col_sum:
        if uploaded_files and st.button("📝 Summarize PDF(s)"):
            # Concatenate all docs' text for summary
            docs = load_pdfs_from_fileobjs(uploaded_files)
            all_text = "\n".join([doc.page_content for doc in docs])
            summary_prompt = f"Summarize the following document(s):\n{all_text[:4000]}"  # Limit for token safety
            with st.spinner("Summarizing..."):
                summary = llm(summary_prompt)
            st.markdown(f'<div class="chat-bubble"><b>Summary:</b> {summary}</div>', unsafe_allow_html=True)
# Show chat history with anchors and references
if st.session_state.chat_history:
    for idx, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        anchor = f"q{len(st.session_state.chat_history)-idx}"
        st.markdown(f"<a id='{anchor}'></a>", unsafe_allow_html=True)
        st.markdown(f'<div class="user-bubble"><b>You:</b> {q}</div>', unsafe_allow_html=True)
        # --- PDF Page Reference and Citation ---
        # Get relevant docs/pages for this answer
        if uploaded_files:
            # Use retriever to get source docs (top 2 shown as example)
            relevant_docs = db.similarity_search(q, k=2)
            refs = ", ".join([f"Page {doc.metadata.get('page', '?')}" for doc in relevant_docs])
            st.markdown(f'<div class="chat-bubble"><b>Bot:</b> {a}<br><span style="font-size:0.9em;color:#aaa;">References: {refs}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble"><b>Bot:</b> {a}</div>', unsafe_allow_html=True)
else:
    st.info("Please upload at least one PDF to start chatting.")


st.markdown("</div>", unsafe_allow_html=True)

