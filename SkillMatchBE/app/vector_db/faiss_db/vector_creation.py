import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.config import DATA_PATH,APP_PATH

file_path = DATA_PATH + "/Interview questions.xlsx"

df = pd.read_excel(file_path)
# print(df["Questions"])

interview_questions = list(df['Questions'].unique())

# Wrap each question in a Document
docs = [Document(page_content=q) for q in interview_questions]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs, embedding_model)
vector_store_path = APP_PATH + "/vector_db/vector_store/faiss_index" 
vectorstore.save_local(vector_store_path)

print("FAISS index built and saved.")
