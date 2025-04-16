import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Set up Streamlit UI
st.set_page_config(page_title="Conversational RAG", layout="wide")
st.title("🤖 Conversational RAG with PDF Upload + Chat History")
st.write("Upload a PDF and chat with it using LangChain, Groq, and Hugging Face Embeddings.")

# 🔐 Ask for API keys securely
groq_api_key = st.text_input("🔑 Enter your Groq API Key:", type="password")
hf_token = st.text_input("🧠 Enter your Hugging Face Token:", type="password")

# ✅ Proceed only if both keys are provided
if groq_api_key and hf_token:
    # Set Hugging Face token for embedding
    os.environ['HF_TOKEN'] = hf_token
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

    # Chat session
    session_id = st.text_input("💬 Session ID:", value="default_session")

    # Initialize session storage for chat history
    if "store" not in st.session_state:
        st.session_state.store = {}

    # File upload
    uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf", accept_multiple_files=False)

    if uploaded_file:
        # Save PDF temporarily
        temp_path = "./temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Split and embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Contextual question prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Given a chat history and the latest user question "
             "which might reference context in the chat history, "
             "formulate a standalone question which can be understood "
             "without the chat history. Do NOT answer the question, "
             "just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # Answering prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an assistant for question-answering tasks. "
             "Use the following pieces of retrieved context to answer "
             "the question. If you don't know the answer, say that you "
             "don't know. Use three sentences maximum and keep the "
             "answer concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Chat history retrieval
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # User input
        user_question = st.text_input("❓ Ask a question about your PDF:")
        if user_question:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_question},
                config={"configurable": {"session_id": session_id}},
            )
            st.markdown("**Assistant:** " + response["answer"])
            with st.expander("🕒 Chat History"):
                for msg in session_history.messages:
                    st.write(msg)
else:
    st.info("Please enter your Groq and Hugging Face API keys to begin.")
