🤖 Conversational RAG PDF Chatbot with Groq, LangChain & Hugging Face

This project is a powerful Conversational RAG (Retrieval-Augmented Generation) application built with Streamlit, enabling users to upload a PDF file and engage in a dynamic, context-aware Q&A conversation based on the document content. The chatbot leverages Groq's blazing-fast inference (Gemma2-9b) for language understanding and Hugging Face's MiniLM embeddings to encode and search the document content using a FAISS vector store.

Unlike simple document Q&A systems, this app supports multi-turn conversation with memory, meaning it understands follow-up questions by maintaining and referencing chat history context. A background chain reformulates vague or referential queries into standalone questions before retrieving information from the uploaded document. This makes the interaction feel much more like a natural and intelligent conversation.

Key components:

->🔍 PDF Upload and automatic chunking using RecursiveCharacterTextSplitter

->💾 FAISS-based vector store for fast document retrieval

->🧠 Chat-aware question reformulation using LangChain's message history integration

->✨ Groq + Gemma2 LLM for fast, high-quality generation

->🗂️ Session-based chat history managed in Streamlit for each session

The app is ideal for:

->Researchers who want to explore and question academic papers

->Students or educators reviewing dense PDFs and looking for key answers

->Developers and AI enthusiasts experimenting with RAG architectures and conversational memory

->The included sample was tested using the "Attention Is All You Need" paper, but the system is compatible with any PDF you upload.
