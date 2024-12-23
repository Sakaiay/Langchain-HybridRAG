import os
import uuid
import openai
import chromadb
from splitter.text_splitter import TextSplit
from embedding.local_embedding import LocalEmbedding
from chromadb.config import Settings
from langchain_chroma import Chroma

class ChromaDatabase:
    '''
    自定义chromabd，将文本添加进知识库，在知识库中删除。
    '''
    def __init__(self, db_name: str, embedding):
         # 初始化数据库名称和存储目录
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.db_directory = os.path.join(self.current_directory, "chroma_db")
        self.db_name = db_name
        self.embedding = embedding
        self.client = chromadb.PersistentClient(path=self.db_directory)
        self.collection = self.client.get_or_create_collection(name=self.db_name)
        
        self.chroma = Chroma(
            client=self.client,
            collection_name=self.db_name,
            embedding_function=self.embedding,
        )
        
    def get_vectorstore(self):
        return self.chroma
        
    # def add_texts(self, texts: list, metadatas: list = None, ids: list = None):
        
    #     embedding_texts = self.embedding.embed_documents(texts)
    #     # 默认元数据为空
    #     if metadatas is None:
    #         metadatas = [{} for _ in texts]
            
    #     if ids is None:
    #         ids = [str(uuid.uuid4()) for _ in texts]
            
    #     self.collection.add(
    #         documents = texts,
    #         metadatas = metadatas,
    #         embeddings = embedding_texts,
    #         ids = ids
    #     )
        
    # def add_documents(self, 
    #                   documents):
    #     texts = [doc.page_content for doc in documents]
    #     metadatas = [doc.metadata for doc in documents]
        
    #     self.add_texts(texts, metadatas)
    
    def add_docs(self, 
                 docs):  
        doc_infos = []  
        texts = [doc.page_content for doc in docs]  
        metadatas = [doc.metadata for doc in docs]  
        embeddings = self.embedding.embed_documents(texts=texts)  
        ids = [str(uuid.uuid1()) for _ in range(len(texts))]  
        for _id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):  
            self.chroma._collection.add(  
                ids=_id, embeddings=embedding, metadatas=metadata, documents=text  
            )  
            doc_infos.append({"id": _id, "metadata": metadata})  
        return doc_infos  
    
    
        
if __name__ == '__main__':
    # file_path = '/data01/tqbian/src/learning/RAG/Langchain-HybridRAG/data/人物介绍.txt'
    # tp = TextSplit(file_path)
    # texts = tp.split_text()
    
    # _params = {
    #     "base_url": 'http://127.0.0.1:9997/v1',
    #     "api_key": 'EMPTY',
    # }
    # client = openai.Client(**_params)
    # embeddings = LocalEmbedding(model='bge-m3', client=client)
    # chroma_db = ChromaDatabase(db_name='test', embedding=embeddings)
    
    # chroma_db.add_documents(texts)
    
    print('bbb')
    
    
    
    
        
        
        
        
        
        
    
    
    
    
    
    
    
    