from typing import List, Any
from sentence_transformers import SentenceTransformer


class SearchTool:
    def __init__(self, index,vindex = None, embedding_model="multi-qa-MiniLM-L6-cos-v1"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = index
        self.vindex = vindex

    def text_search(self, query: str) -> List[Any]:
        """
        Perform a text-based search on the FAQ index.

        Args:
            query (str): The search query string.

        Returns:
            List[Any]: A list of up to 5 search results returned by the FAQ index.
        """
        return self.index.search(query, num_results=5)
    
    def vector_search(self, query: str) -> List[Any]:
        """
        Perform a vector-based search on the FAQ index.

        Args:
            query (str): The search query string.
        Returns:
            List[Any]: A list of up to 5 search results returned by the FAQ index.
        """
        q = self.embedding_model.encode(query)
        return self.vindex.search(q, num_results=5)
    
    def hybrid_search (self, query: str) -> List[Any]:
        """
        Perform a hybrid search (text + keyword) on the FAQ index.

        Args:
            query (str): The search query string.

        Returns:
            List[Any]: A list of up to 5 search results returned by the FAQ index.
        """
        text_results = self.text_search(query)
        vector_results = self.vector_search(query)

        seed_ids = set()
        combined_results = []

        for result in text_results + vector_results:
            if result["filename"] not in seed_ids:
                seed_ids.add(result["filename"])
                combined_results.append(result)

        return combined_results[:5]
        