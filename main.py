import os
import openai
import utils.cprint as ct 
from utils import logger, prompt_template
from splitter.text_splitter import TextSplit
from splitter.sim_splitter import SimilarityBasedSplitter
from langchain_community.document_loaders import TextLoader
from embedding.local_embedding import LocalEmbedding
from knowledge_db.chroma_service import ChromaDatabase
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from retriever.retriever_service import RetireverService


if __name__ == '__main__':
    
    # 定义embedding模型
    _params = {
        "base_url": 'http://127.0.0.1:9997/v1',
        "api_key": 'EMPTY',
    }
    client = openai.Client(**_params)
    embeddings = LocalEmbedding(model='bge-m3', client=client)
    
    # 定义LLM
    params = {
        "model_name" : "qwen2-instruct",
        "temperature": 0.2,
        "base_url": 'http://127.0.0.1:9997/v1',
        "api_key": 'EMPTY',
    }
    llm = ChatOpenAI(**params)
    
    
    # 分割文本
    file_path = './data/人物介绍.txt'
    loader = TextLoader(file_path) 
    documents = loader.load()
    
    splitter = SimilarityBasedSplitter(embeddings)
    texts = splitter.split_docs(documents)
    
    # splitter = TextSplit(chunk_size=250, chunk_overlap=150)
    # texts = splitter.split_docs(documents)
    

    chroma_db = ChromaDatabase(db_name='db_test', embedding=embeddings)
    
    doc_info = chroma_db.add_docs(texts)
    
    retriever = RetireverService(top_k=5, score_threshold=0.4, vectorstore=chroma_db.get_vectorstore())
    
    prompt_ = prompt_template.rag_prompt
    query = "特朗普的名言"
    retriever_docs = retriever.get_relevant_documents(query)
    print(retriever_docs)
    prompt = ChatPromptTemplate.from_messages([prompt_])
    context = "\n\n".join([doc.page_content for doc in retriever_docs])
    chain = prompt | llm
    res = chain.invoke({"context": context, "question": query})
    print(res.content)

    