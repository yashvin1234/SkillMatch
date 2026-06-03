from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStore
from typing import List, Dict

class QuestionSuggester:
    def __init__(self, model_name: str, faiss_path: str):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore: VectorStore = FAISS.load_local(
            faiss_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

    def suggest_questions(self, input_text: str, top_k: int = 15, filter_value: str = "question") -> List[str]:
        results = self.vectorstore.similarity_search_with_score(input_text, k=top_k, filter={"type": filter_value})
        return [doc.page_content for doc, _ in results]

    def suggest_courses(self, input_text: str, top_k: int = 15, filter_value: str = "resource") -> List[Dict]:
        results = self.vectorstore.similarity_search_with_score(input_text, k=top_k, filter={"type": filter_value})
        return [
            {
                "course":       doc.metadata.get("resource", ""),       # Pathway Display Name
                "category":     doc.metadata.get("skill_area", ""),     # Category
                "topic":        doc.metadata.get("sub_skill", ""),      # Skill/Topic Pathways
                "level":        doc.metadata.get("course_level", ""),   # Course Level
                "url":          doc.metadata.get("url", ""),            # Pathway URL
            }
            for doc, _ in results
        ]