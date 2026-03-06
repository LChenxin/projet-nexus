# backend/tools/internal_search.py

import json
from typing import List, Dict, Any, Tuple
import numpy as np

from backend import config
from backend.llm import embed


# =========================
# 1) Similarity search
# =========================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =========================
# 2) InternalSearch 
# =========================
class InternalSearch:

    def __init__(self):
        with open(config.INTERNAL_PROJECTS_FILE, "r", encoding="utf-8") as f:
            self.projects: List[Dict[str, Any]] = json.load(f)

        self.texts = [self._build_text(p) for p in self.projects]

        print("Building internal embeddings...")
        vectors = embed(self.texts)
        self.embeddings = np.array(vectors)


    def _build_text(self, project: Dict[str, Any]) -> str:
        parts = [
            project.get("title", ""),
            project.get("objective", ""),
            project.get("outcome", ""),
            " ".join(project.get("lessons_learned", [])),
            " ".join(project.get("tech_stack", [])),
        ]
        return " ".join(parts)



    def search(self, query: str, allow_c2: bool = False) -> Dict[str, Any]:
            
            query_vec = np.array(embed([query])[0])
            
            query_norm = query_vec / np.linalg.norm(query_vec)
            docs_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            scores = np.dot(docs_norm, query_norm)

            results: List[Tuple[float, Dict[str, Any]]] = []

            for idx, project in enumerate(self.projects):
                
                if not allow_c2 and project.get("confidentiality") == "C2":
                    continue

                score = scores[idx]
                if score >= config.RETRIEVAL_THRESHOLD:
                    results.append((score, project))

            results.sort(key=lambda x: x[0], reverse=True)
            top_results = results[:config.RETRIEVAL_TOP_K]

            evidence = []
            for score, project in top_results:
                evidence.append({
                    "project_id": project["project_id"],
                    "title": project["title"],
                    "score": round(float(score), 4),
                    "summary": project["objective"],
                    "status": project["status"],
                    "confidentiality": project.get("confidentiality", "Public") 
                })

            return {
                "query": query,
                "results": evidence,
                "count": len(evidence),
            }


# =========================
# 3) instnace
# =========================
_internal_search_instance = None


def get_internal_search() -> InternalSearch:
    global _internal_search_instance
    if _internal_search_instance is None:
        _internal_search_instance = InternalSearch()
    return _internal_search_instance