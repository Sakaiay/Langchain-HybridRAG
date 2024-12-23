from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplit():
    '''
    文本切分器，用于返回切分后的文档
    '''
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separators=["。", "？", "！"], add_start_index=True)
    
    def split_docs(self, documents):
        '''
        切分文档
        '''
        texts = self.splitter.split_documents(documents)
        return texts
    
