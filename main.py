import os
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

st.title("🤖 Local AI PDF Workspace")
st.write("Secure offline analyzer featuring real-time stream generation.")


# Initialize the model engine
@st.cache_resource
def load_llm():
    return ChatOllama(model="llama3.2", temperature=0, num_ctx=8192)


try:
    llm = load_llm()
except Exception as e:
    st.error(f"Ollama Connection Error: {e}")

# Sidebar Document Center
with st.sidebar:
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader("Upload your document.pdf here", type=["pdf"])

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
    if user_question := st.chat_input("Ask a question about this document..."):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Process and stream the response content live on screen
        with st.chat_message("assistant"):
            prompt = f"""
            You are a data assistant analyzing a PDF. Use the extracted text to answer cleanly.
            ---
            EXTRACTED CONTENT:
            {st.session_state.get("pdf_text", "")}
            ---
            QUESTION: {user_question}
            """

            # Use st.write_stream to draw the characters out word-by-word
            def response_generator():
                for chunk in llm.stream(prompt):
                    yield chunk.content

            ai_answer = st.write_stream(response_generator())

        st.session_state.messages.append({"role": "assistant", "content": ai_answer})

        # Log conversation cleanly to file
        with open("ai_analysis_report.txt", "a", encoding="utf-8") as report_file:
            report_file.write(f"Question: {user_question}\nAnswer:\n{ai_answer}\n\n")

else:
    st.info(
        "💡 Please drag and drop a PDF file into the left sidebar to activate the AI conversational layer."
    )
