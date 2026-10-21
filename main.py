import os
import subprocess
import streamlit as st
from langchain_ollama import ChatOllama
from pdf2image import convert_from_path
import pytesseract

# Page Configuration Setup
st.set_page_config(
    page_title="Local AI Workspace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Local AI PDF & Coding Workspace")
st.write("Secure offline analyzer featuring real-time stream generation.")


# Helper function to get local git changes (git diff)
def get_git_changes():
    try:
        # Run 'git diff' to see what changed in the files
        result = subprocess.run(
            ["git", "diff", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return ""


# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Workspace Controls")

    # 1. Dynamic Model Selection Dropdown
    selected_model = st.selectbox(
        "Select Local AI Model:",
        options=["llama3.2", "qwen2.5-coder:1.5b"],
        index=0,
        help="Choose llama for general documents and qwen for coding tasks.",
    )

    st.markdown("---")
    st.header("💻 Git Assistant Tools")

    # 2. Smart Commit Message Button
    if st.button("🤖 Generate Smart Commit", use_container_width=True):
        changes = get_git_changes()
        if not changes:
            st.sidebar.info("✨ No unstaged code changes found. Everything is clean!")
        else:
            # Initialize Qwen specifically for the code change task
            commit_llm = ChatOllama(model="qwen2.5-coder:1.5b", temperature=0)
            commit_prompt = f"""
            Task: Write a short, one-sentence Git commit message for the code changes below.
            
            Strict Rules:
            - Start with a clear action verb (e.g., 'Add', 'Fix', 'Refactor', 'Update').
            - Output ONLY the raw text sentence.
            - Do NOT wrap it in quotes, code blocks, or markdown.
            - Do NOT explain anything or print instructions.

            --- CODE CHANGES ---
            {changes}
            --------------------
            """
            with st.sidebar.spinner("Analyzing changes..."):
                commit_response = commit_llm.invoke(commit_prompt)
                st.sidebar.success("💡 Suggested Commit Message:")
                st.sidebar.code(commit_response.content, language="text")

    st.markdown("---")
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader("Upload your document.pdf here", type=["pdf"])

    st.markdown("---")
    # 3. Clear Chat History Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.toast("Conversation memory cleared!")
        st.rerun()


# Initialize the selected model engine dynamically
@st.cache_resource
def load_llm(model_name):
    return ChatOllama(model=model_name, temperature=0, num_ctx=8192)


try:
    llm = load_llm(selected_model)
except Exception as e:
    st.error(f"Ollama Connection Error: {e}")

# Main Application Logic
if uploaded_file is not None:
    temp_filename = "temp_uploaded_document.pdf"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process text via OCR
    if "pdf_text" not in st.session_state:
        with st.spinner("👁️ Running OCR Text Extraction... Please wait..."):
            try:
                pages = convert_from_path(temp_filename, 150)
                extracted_text = ""
                for page_image in pages:
                    extracted_text += pytesseract.image_to_string(page_image) + "\n"

                st.session_state["pdf_text"] = extracted_text
                st.sidebar.success(f"Loaded {len(pages)} pages successfully!")
            except Exception as e:
                st.error(f"OCR Error: {e}")
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

    # Initialize chat history memory states
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Print out previous chat conversation cards cleanly
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept new user text prompt input entries
    if user_question := st.chat_input(
        "Ask a question about this document or code task..."
    ):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Process and stream the response content live on screen
        with st.chat_message("assistant"):
            prompt = f"""
            You are an advanced local AI assistant. Use the context below if provided to answer the question cleanly.
            ---
            CONTEXT DATA:
            {st.session_state.get("pdf_text", "")}
            ---
            QUESTION: {user_question}
            """

            def response_generator():
                for chunk in llm.stream(prompt):
                    yield chunk.content

            ai_answer = st.write_stream(response_generator())

        st.session_state.messages.append({"role": "assistant", "content": ai_answer})

        # Log conversation cleanly to file
        with open("ai_analysis_report.txt", "a", encoding="utf-8") as report_file:
            report_file.write(
                f"[{selected_model}] Question: {user_question}\nAnswer:\n{ai_answer}\n\n"
            )

else:
    st.info(
        "💡 Please drag and drop a PDF file into the left sidebar to activate the AI conversational layer."
    )
    st.info(
        "💡 Note: You can switch models in the sidebar menu at any time depending on your task."
    )
