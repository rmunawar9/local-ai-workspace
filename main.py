import os
import subprocess
import streamlit as st
from langchain_ollama import ChatOllama
from pdf2image import convert_from_path
import pytesseract
from markitdown import MarkItDown

# Page Configuration Setup
st.set_page_config(
    page_title="Local AI Workspace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Local AI Multi-Tool Workspace")
st.write("Secure offline analyzer, multi-document converter, and coding engine.")


# Helper function to get local git changes
def get_git_changes():
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return ""


# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Workspace Controls")

    selected_model = st.selectbox(
        "Select Local AI Model:",
        options=["llama3.2", "qwen2.5-coder:1.5b"],
        index=0,
        help="Choose llama for general documents and qwen for coding tasks.",
    )

    st.markdown("---")
    st.header("💻 Git Assistant Tools")

    if st.button("🤖 Generate Smart Commit", use_container_width=True):
        changes = get_git_changes()
        if not changes:
            st.sidebar.info("✨ No unstaged code changes found. Everything is clean!")
        else:
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
            Commit Message:"""
            with st.sidebar.spinner("Analyzing changes..."):
                commit_response = commit_llm.invoke(commit_prompt)
                st.sidebar.success("💡 Suggested Commit Message:")
                st.sidebar.code(commit_response.content, language="text")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pdf_text", None)  # Reset extracted context
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

# Build clean horizontal workspace layout tabs
tab1, tab2 = st.tabs(["💬 Local AI Document Chat", "📝 File to Markdown Converter"])

# ==================== TAB 1: AI DOCUMENT CHAT ====================
with tab1:
    st.header("📂 Chat with Scanned PDFs")
    # Added accept_multiple_files=True here
    uploaded_files = st.file_uploader(
        "Upload your document.pdf here:",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_chat_uploader",
    )

    if uploaded_files:
        if "pdf_text" not in st.session_state:
            combined_text = ""
            for uploaded_file in uploaded_files:
                temp_filename = f"temp_{uploaded_file.name}"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"👁️ Running OCR Extraction on {uploaded_file.name}..."):
                    try:
                        pages = convert_from_path(temp_filename, 150)
                        for page_image in pages:
                            combined_text += (
                                pytesseract.image_to_string(page_image) + "\n"
                            )
                    except Exception as e:
                        st.error(f"OCR Error on {uploaded_file.name}: {e}")
                    finally:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)

            st.session_state["pdf_text"] = combined_text
            st.success("✅ All uploaded documents processed and merged into AI memory!")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_question := st.chat_input(
            "Ask a question about your uploaded documents..."
        ):
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

            with st.chat_message("assistant"):
                prompt = f"""
                You are a data assistant analyzing document context. Use the extracted text below to answer cleanly.
                ---
                EXTRACTED CONTENT:
                {st.session_state.get("pdf_text", "")}
                ---
                QUESTION: {user_question}
                """

                def response_generator():
                    for chunk in llm.stream(prompt):
                        yield chunk.content

                ai_answer = st.write_stream(response_generator())

            st.session_state.messages.append(
                {"role": "assistant", "content": ai_answer}
            )

            with open("ai_analysis_report.txt", "a", encoding="utf-8") as report_file:
                report_file.write(
                    f"Question: {user_question}\nAnswer:\n{ai_answer}\n\n"
                )
    else:
        st.info(
            "💡 Drop one or multiple PDF files here to activate the conversational chat box."
        )

# ==================== TAB 2: MARKDOWN CONVERTER ====================
with tab2:
    st.header("🔄 Universal Markdown Converter")
    st.write("Convert multiple files simultaneously into structured `.md` format.")

    # Added accept_multiple_files=True here
    convert_files = st.file_uploader(
        "Upload Office or Text documents:",
        type=["pdf", "docx", "xlsx", "pptx", "txt", "html"],
        accept_multiple_files=True,
        key="md_converter_uploader",
    )

    if convert_files:
        md_engine = MarkItDown()

        for file in convert_files:
            file_ext = file.name.split(".")[-1]
            temp_conv_name = f"temp_convert_{file.name}"

            with open(temp_conv_name, "wb") as f:
                f.write(file.getbuffer())

            with st.spinner(f"Converting {file.name} to Markdown..."):
                try:
                    md_result = md_engine.convert(temp_conv_name)
                    markdown_output = md_result.text_content

                    # Create a nice layout expander window for each converted file
                    with st.expander(f"📄 Preview & Download: {file.name}"):
                        st.text_area(
                            "Markdown Code",
                            value=markdown_output,
                            height=150,
                            key=f"txt_{file.name}",
                        )

                        clean_name = file.name.rsplit(".", 1)[0] + ".md"
                        st.download_button(
                            label=f"💾 Download {clean_name}",
                            data=markdown_output,
                            file_name=clean_name,
                            mime="text/markdown",
                            key=f"btn_{file.name}",
                        )
                except Exception as conv_err:
                    st.error(f"Error converting {file.name}: {conv_err}")
                finally:
                    if os.path.exists(temp_conv_name):
                        os.remove(temp_conv_name)
