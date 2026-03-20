import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.config import DATA_PATH, APP_PATH

# Load interview questions
All_Data:pd.DataFrame = pd.read_excel(DATA_PATH +"/Interview MasterData.xlsx",sheet_name=None)
questions_df= All_Data['Questions']

interview_questions = list(questions_df['questions'].dropna().unique())
question_docs = [Document(page_content=q, metadata={"type": "question"}) for q in interview_questions]

# Load skills and resources data
skills_df = All_Data['Upskilling Training Plan']  # Change to your actual file name


questions_df = pd.read_excel(DATA_PATH + "/Interview questions.xlsx")
interview_questions = list(questions_df['Questions'].dropna().unique())
question_docs = [Document(page_content=q, metadata={"type": "question"}) for q in interview_questions]

# Load skills and resources data
skills_df = pd.read_excel(DATA_PATH + "/Skills_Resources.xlsx")  # Change to your actual file name

skill_docs = []
for _, row in skills_df.iterrows():
    content = f"""
    Skill Area: {row['Skill Area']}
    Sub-Skill: {row['Sub-Skill/Topic']}
    Learning Objectives: {row['Learning Objectives']}
    Recommended Resource: {row['Recommended Resources']}
    Duration: {row['Duration']}
    """
    skill_docs.append(
        Document(
            page_content=content.strip(),
            metadata={
                "type": "resource",
                "skill_area": row["Skill Area"],
                "sub_skill": row["Sub-Skill/Topic"],
                "resource": row["Recommended Resources"],
                "duration": row["Duration"]
            }
        )
    )

# Combine all documents
all_docs = question_docs + skill_docs

# Create embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build vector store
vectorstore = FAISS.from_documents(all_docs, embedding_model)

# Save vector store
vector_store_path =APP_PATH + "vector_db/vector_store/faiss_index2"
vectorstore.save_local(vector_store_path)

print("FAISS index with questions and skill resources built and saved.")
