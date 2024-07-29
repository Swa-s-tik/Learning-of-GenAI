import sqlite3
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

# Database setup
def setup_database():
    conn = sqlite3.connect('question_papers.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS papers
                 (id INTEGER PRIMARY KEY, 
                  university TEXT, 
                  subject TEXT, 
                  year INTEGER, 
                  content TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS generated_questions
                 (id INTEGER PRIMARY KEY, 
                  university TEXT, 
                  subject TEXT, 
                  question TEXT)''')
    conn.commit()
    return conn

# Document Processing
def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Embedding and Storage
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM Setup (adjust the path to where you've stored the LLaMA model)
llm = LlamaCpp(model_path="./models/llama-7b.ggmlv3.q4_0.bin")

# Main processing function
def process_and_store_paper(pdf_path, university, subject, year, conn):
    # Process PDF
    text = process_pdf(pdf_path)
    
    # Store in SQLite
    c = conn.cursor()
    c.execute("INSERT INTO papers (university, subject, year, content) VALUES (?, ?, ?, ?)", 
              (university, subject, year, text))
    conn.commit()
    
    # Split and embed
    chunks = text_splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=[{"university": university, "subject": subject, "year": year} for _ in chunks])
    
    return vectorstore

# Question Generation
def generate_question(vectorstore, university, subject):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"filter": {"university": university, "subject": subject}}
        )
    )
    
    prompt = f"""Based on the context provided for {university}'s {subject} course, 
    generate a multiple-choice question in the style of a university exam. 
    Include 4 options and indicate the correct answer."""
    
    result = qa_chain({"query": prompt})
    return result["result"]

# Main execution
def main():
    conn = setup_database()
    
    # Process all PDFs in a directory
    pdf_dir = Path("path/to/your/pdfs")
    vectorstores = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        # In a real scenario, you'd extract these details from the filename or a metadata file
        university = "Example University"
        subject = "Computer Science"
        year = 2023
        vectorstore = process_and_store_paper(pdf_path, university, subject, year, conn)
        vectorstores.append(vectorstore)
    
    # Combine all vectorstores
    combined_vectorstore = vectorstores[0]
    for vs in vectorstores[1:]:
        combined_vectorstore.merge_from(vs)
    
    # Generate questions for a specific university and subject
    university = "Example University"
    subject = "Computer Science"
    for _ in range(10):  # Generate 10 questions
        question = generate_question(combined_vectorstore, university, subject)
        c = conn.cursor()
        c.execute("INSERT INTO generated_questions (university, subject, question) VALUES (?, ?, ?)", 
                  (university, subject, question))
        conn.commit()
        print(question)
    
    conn.close()

if __name__ == "__main__":
    main()