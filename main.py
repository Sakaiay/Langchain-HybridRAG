import os
import openai
import utils.cprint as ct 
from config import get_config
from utils import logger, prompt_template
from splitter import *
from langchain_community.document_loaders import TextLoader
from embedding.local_embedding import LocalEmbedding
from knowledge_db.chroma_service import ChromaDatabase
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from retriever.retriever_service import RetireverService


if __name__ == '__main__':
    config = get_config()
    
    llm_config = config['llm']
    embeddings_config = config['embedding']
    
    # 定义embedding模型
    embedding_name = embeddings_config['model']
    embedding_params = embeddings_config['params']
    embeddings = LocalEmbedding(model=embedding_name, client=openai.Client(**embedding_params))
    
    # 定义LLM
    llm = ChatOpenAI(**llm_config['params'])
    
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
    # print(retriever_docs)
    prompt = ChatPromptTemplate.from_messages([prompt_])
    context = "\n\n".join([doc.page_content for doc in retriever_docs])
    chain = prompt | llm
    res = chain.invoke({"context": context, "question": query})
    print(res.content)

    