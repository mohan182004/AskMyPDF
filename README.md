
# AskMyPDF

**AskMyPDF** is a user-friendly web application that allows you to upload one or more PDF files and interact with their content using natural language. Ask questions about your PDFs, get instant answers, and even generate summaries—all powered by advanced language models.

---

## Features

- **Chat with Your PDFs:** Upload PDF files and ask questions about their content in plain English.
- **Instant Answers:** Receive accurate responses with page references from your PDFs.
- **Summarization:** Generate concise summaries of your uploaded documents.
- **Chat History:** Review your previous questions and answers in a sidebar.
- **Modern UI:** Enjoy a clean, chat-like interface built with Streamlit.
- **Multiple PDF Support:** Upload and interact with multiple PDF files at once.

---

## How It Works

1. **Upload PDFs:** Use the file uploader to add one or more PDF files.
2. **Ask Questions:** Type your questions in the chat input. The app finds the answer from your PDFs, citing relevant pages.
3. **Summarize Documents:** Generate a summary of your uploaded files with a single click.
4. **Clear Session:** Easily clear your uploaded files and chat history.

---

## Technologies Used

- **Python**
- **Streamlit** (UI)
- **LangChain** (Conversational retrieval, PDF parsing)
- **HuggingFace Transformers** (Embeddings)
- **OpenAI GPT-4.1** (LLM backend, customizable in `custom_llm.py`)
- **PyPDF2** (PDF reading)

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mohan182004/AskMyPDF.git
   cd AskMyPDF
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables:**  
   Create a `.env` file with your LLM API keys and URLs (see code for variable names).
4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## Usage

- Upload one or more PDF files.
- Enter your questions in the input box and press Enter.
- To summarize PDFs, use the "Summarize PDF(s)" button.
- Clear session using the "Clear PDFs & Chat" button.

---

## Example Use Cases

- **Research:** Quickly extract answers from research papers.
- **Study:** Summarize textbooks or lecture notes.
- **Business:** Analyze reports, contracts, or manuals.


## License

This project currently does not have a license specified.


            
