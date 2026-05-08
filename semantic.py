from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

def semantic_chunking(document,chunk_size=800,chunk_overlap=200):
    #splitting the semantic chunks
    text_splitting=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',' ',''],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    semantic_split=text_splitting.split_text(document)
    #regex pattern for identifying the heading
    semantic_pattern=[r'^#+\s+(.+)$',r'']
