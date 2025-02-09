# 🧠 MindMate+ 🤖

A compassionate AI-powered chatbot designed to assist with mental well-being by answering questions and providing support based on uploaded PDFs.

> ⚠ **Disclaimer:** This chatbot is not a substitute for professional medical advice. If you are experiencing a crisis, please seek help from a licensed professional.

## 🚀 Features
- 💬 **Conversational AI:** Uses `llama-3.3-70b-versatile` from **Groq**.
- 📄 **Document-based Retrieval:** Upload PDFs to enhance responses.
- 🔎 **Vector Search with ChromaDB:** Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- 🌐 **User-Friendly Interface:** Built with **Gradio** for a seamless experience.

## 🛠 Tech Stack

### **Backend**
- **Python** (Core programming language)
- **LangChain** (LLM-based application framework)
- **ChatGroq** (Groq API for LLM)
- **Hugging Face BGE Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`)
- **ChromaDB** (Vector database for document retrieval)

### **Frontend**
- **Gradio** (Web UI framework)

### **Document Processing**
- **PyPDFLoader** (Extracts text from PDFs)
- **DirectoryLoader** (Loads multiple PDFs)
- **RecursiveCharacterTextSplitter** (Splits text into chunks for embedding)

## 📦 Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/mental-health-chatbot.git
cd mental-health-chatbot
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Set Up Groq API Key
Replace `groq_api_key` in `initialize_llm()` with your own API key.

```python
groq_api_key="your_api_key_here"
```

### 4️⃣ Run the Application
```sh
python chatbot.py
```

The chatbot will be available at `http://localhost:7860/`

## 📄 Usage
- Upload PDFs related to mental health (optional)
- Ask questions via the chatbot interface
- Receive AI-powered responses based on documents

## 🔥 Contributing
Feel free to fork this repository and submit a pull request with enhancements!



---


