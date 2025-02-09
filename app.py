from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gradio as gr

def initialize_llm():
    from langchain_groq import ChatGroq
    
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )

    return llm

def create_vector_db():
    # Using the current directory (./) since the PDF is uploaded to the repository root.
    loader = DirectoryLoader("./", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    
    print("ChromaDB created and data saved")
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot:"""
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

print("Initializing Chatbot.........")
llm = initialize_llm()

# Use a relative path for the vector database directory.
db_path = "./chroma_db"

if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
qa_chain = setup_qa_chain(vector_db, llm)

def chatbot_response(user_input, history):
    if not user_input.strip():
        return {"role": "assistant", "content": "Please provide a valid input."}
    assistant_response = qa_chain.run(user_input)
    return {"role": "assistant", "content": assistant_response}

with gr.Blocks(theme='Respair/Shiki@1.2.1') as app:
    gr.Markdown("# 🧠 Mental Health Chatbot 🤖")
    gr.Markdown("A compassionate chatbot designed to assist with mental well-being. Please note: For serious concerns, contact a professional.")
    
    chatbot = gr.ChatInterface(
        fn=chatbot_response, 
        title="Mental Health Chatbot", 
        type="messages"
    )
    
    gr.Markdown("This chatbot provides general support. For urgent issues, seek help from licensed professionals.")

app.launch()
