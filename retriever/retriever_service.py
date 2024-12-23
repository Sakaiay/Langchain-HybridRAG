from langchain_core.retrievers import BaseRetriever

class RetireverService():
    def __init__(self,
                top_k,
                score_threshold,
                vectorstore,
                ) -> None:
        self.top_k = top_k
        self.vectorstore = vectorstore
        self.score_threshold = score_threshold
        
        self.get_retirever()
    
    def get_retirever(self):
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.top_k})
        # self.retriever = self.vectorstore.as_retriever(
        #     search_type="similarity_score_threshold",
        #     search_kwargs={"score_threshold": self.score_threshold, "k": self.top_k},
        # )
    
    def get_relevant_documents(self, query):
        return self.retriever.invoke(query)
        # return self.retriever.get_relevant_documents(query)[: self.top_k]
    
    