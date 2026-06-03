import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

skills_df = pd.read_excel(os.path.join(BASE_DIR, "Skills_Resources.xlsx"))

skill_docs = []
for _, row in skills_df.iterrows():
    content = f"""
Category: {row['Category']}
Skill/Topic: {row['Skill/Topic Pathways']}
Description: {row['Description']}
Course: {row['Pathway Display Name']}
Level: {row['Course Level']}
URL: {row['Pathway URL']}
    """
    skill_docs.append(
        Document(
            page_content=content.strip(),
            metadata={
                "type": "resource",
                "skill_area": row["Category"],
                "sub_skill": row["Skill/Topic Pathways"],
                "resource": row["Pathway Display Name"],
                "course_level": row["Course Level"],
                "url": row["Pathway URL"],
            }
        )
    )

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(skill_docs, embedding_model)

vector_store_path = os.path.join(BASE_DIR, "..", "vector_store", "faiss_index2")
vectorstore.save_local(vector_store_path)

print(f"Reindexed {len(skill_docs)} course documents saved to faiss_index2.")