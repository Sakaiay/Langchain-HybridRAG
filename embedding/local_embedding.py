import openai
from langchain.embeddings.base import Embeddings
from pydantic import Field, BaseModel


class LocalEmbedding(Embeddings):
    '''
    自定义embedding模型，用于文本向量化
    '''  
    def __init__(self, 
                 model:str = Field(default='bge-m3', description="使用的嵌入模型名称"), 
                 client:object = Field(default=None, description=""),
                 ):
        
        self.model = model
        self.client = client
        
    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings
    
    def embed_query(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        embeddings = response.data[0].embedding
        return embeddings

if __name__ == '__main__':
    _params = {
        "base_url": 'http://127.0.0.1:9997/v1',
        "api_key": 'EMPTY',
    }
    client = openai.Client(**_params)
    embeddings = LocalEmbedding(model='bge-m3', client=client)
    documents = [
    "这是第一个文档的内容。",
    "这是第二个文档的内容。",
    "第三个文档也包含一些有用的信息。"
    ]
    doc_embeddings = embeddings.embed_documents(documents)
    print("文档嵌入向量:")
    for i, emb in enumerate(doc_embeddings):
        print(f"文档 {i+1} 嵌入向量长度: {len(emb)}")
    query = "什么是检索增强生成（RAG）？"
    query_embedding = embeddings.embed_query(query)
    print("\n查询嵌入向量:")
    print(f"查询嵌入向量长度: {len(query_embedding)}")


        
        