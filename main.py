import os
import subprocess
import streamlit as st
from langchain_ollama import ChatOllama
from pdf2image import convert_from_path
import pytesseract
from markitdown import MarkItDown
from langchain_tavily import TavilySearch
from search_utils import execute_fallback_search

st.set_page_config(
    page_title="Local AI Workspace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Local AI Multi-Tool Workspace")
st.write("Secure offline analyzer, adaptable web search assistant, and coding engine.")

if "uploader_key_pdf" not in st.session_state:
    st.session_state["uploader_key_pdf"] = 100
if "uploader_key_md" not in st.session_state:
    st.session_state["uploader_key_md"] = 200
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar panel architecture controls
with st.sidebar:
    st.header("⚙️ Workspace Controls")
    selected_model = st.selectbox(
        "Select Local AI Model:", options=["llama3.2", "qwen2.5-coder:1.5b"]
    )

    st.markdown("---")
    st.header("🌐 Web Connections")
    tavily_key = st.text_input("Enter Tavily API Key (Optional):", type="password")
    enable_web = st.toggle("Enable Web Grounding", value=False)

    st.markdown("---")
    if st.button(
        "🗑️ Clear Chat & Files", key="reset_chat_tab", use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.pop("pdf_text", None)
        st.session_state["uploader_key_pdf"] += 1
        st.toast("Cleared!")
        st.rerun()


@st.cache_resource
def load_llm(model_name):
    return ChatOllama(model=model_name, temperature=0, num_ctx=8192)


llm = load_llm(selected_model)
tab1, tab2 = st.tabs(["💬 Local AI Document Chat", "📝 File to Markdown Converter"])

with tab1:
    st.header("📂 Chat with Scanned PDFs")
    uploaded_files = st.file_uploader(
        "Upload document.pdf here:",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_{st.session_state['uploader_key_pdf']}",
    )

    if uploaded_files:
        if "pdf_text" not in st.session_state:
            combined_text = ""
            for f in uploaded_files:
                temp_name = f"temp_{f.name}"
                with open(temp_name, "wb") as buffer:
                    buffer.write(f.getbuffer())
                try:
                    pages = convert_from_path(temp_name, 150)
                    for p in pages:
                        combined_text += pytesseract.image_to_string(p) + "\n"
                except:
                    pass
                finally:
                    if os.path.exists(temp_name):
                        os.remove(temp_name)
            st.session_state["pdf_text"] = combined_text

        chat_scroll_area = st.container(height=500)
        with chat_scroll_area:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if user_question := st.chat_input(
            "Ask a question about your uploaded documents..."
        ):
            with chat_scroll_area:
                with st.chat_message("user"):
                    st.markdown(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

            web_context = ""
            if enable_web:
                if tavily_key.strip():
                    os.environ["TAVILY_API_KEY"] = tavily_key.strip()
                    try:
                        search = TavilySearch(max_results=3)
                        web_context = str(search.invoke({"query": user_question}))
                    except Exception as e:
                        web_context = f"Tavily Error: {e}"
                else:
                    web_context = execute_fallback_search(user_question)
            else:
                web_context = "Web search grounding disabled."

            with chat_scroll_area:
                with st.chat_message("assistant"):
                    prompt = f"Answer question based on context sections.\n\nDOC:\n{st.session_state['pdf_text']}\n\nWEB:\n{web_context}\n\nQ: {user_question}"
                    ai_answer = st.write_stream(llm.stream(prompt))
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_answer}
            )
            st.rerun()

with tab2:
    st.header("🔄 Universal Markdown Converter")
    convert_files = st.file_uploader(
        "Upload Office or Text documents:",
        type=["pdf", "docx", "xlsx", "pptx", "txt", "html"],
        accept_multiple_files=True,
        key=f"md_{st.session_state['uploader_key_md']}",
    )

    if convert_files:
        md_engine = MarkItDown()
        for f in convert_files:
            temp_name = f"temp_conv_{f.name}"
            with open(temp_name, "wb") as buffer:
                buffer.write(f.getbuffer())
            try:
                md_result = md_engine.convert(temp_name)
                output_text = md_result.text_content
                with st.expander(f"📄 Preview: {f.name}"):
                    st.text_area(
                        "Markdown Code",
                        value=output_text,
                        height=150,
                        key=f"t_{f.name}",
                    )
                    b_name, _ = os.path.splitext(f.name)
                    st.download_button(
                        label=f"💾 Download {b_name}.md",
                        data=output_text,
                        file_name=f"{b_name}.md",
                        key=f"b_{f.name}",
                    )
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(temp_name):
                    os.remove(temp_name)
