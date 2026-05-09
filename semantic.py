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
    print(f"Documents split into {len(semantic_split)} semantic chunks")
    #regex pattern for identifying the heading | markdown,underlined,all caps headers
    semantic_pattern=[r'^#+\s+(.+)$',r'^.+\n[=\-]{2,}$',r'^[A-Z\s]+:$']
    #convert to document objects with enhanced metadata
    document=[]
    current_section='Introduction'
    for i,chunk in enumerate(semantic_split):
        chunk_lines=chunk.split('\n')
        for lines in chunk_lines:
            for pattern in semantic_pattern:
                match=re.match(pattern,lines,re.MULTILINE)
                if match:
                    current_section=match.group(0)
                    break
        #calculate the semantic density (ratio of non-stopwords to total words)
        words=re.findall(r'\b\w+\b',chunk.lower())
        stopwords = ['the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'it', 'with', 'as', 'for']
        content_words=[w for w in words if w not in stopwords]
        semantic_density=len(content_words)/max(1,len(words))

        doc=Document(
            page_content=chunk,
            metadata={
                'chunk_id':i,
                'total_chunks':len(semantic_split),
                'chunk_size':len(chunk),
                'chunk_type':'semantic',
                'section':current_section,
                'semantic_density':round(semantic_density,2)
            }
        )
        document.append(doc)
    return document

def create_dummy_document():
    """
    Creates a sample multi-section document for testing semantic chunking.
    
    Returns:
        str: A large dummy document with multiple sections
    """
    
    document = """
# INTRODUCTION

Artificial Intelligence (AI) is transforming industries across the world.
Organizations are using machine learning, natural language processing, and
computer vision systems to automate tasks and improve decision-making.

AI systems rely heavily on high-quality data. The better the quality of
training data, the more accurate and reliable the model becomes.
Data preprocessing is therefore one of the most critical stages in an AI pipeline.

---

# DATA COLLECTION

Data collection involves gathering raw information from various sources such as:
- Databases
- APIs
- Sensors
- User interactions
- Documents

Collected data may contain noise, duplicates, or missing values.
Proper validation techniques are required before training machine learning models.

Large enterprises often store their datasets in distributed systems such as
Apache Spark or cloud-based storage solutions.

---

# DATA PREPROCESSING

Data preprocessing includes cleaning and transforming raw data into
a usable format for machine learning algorithms.

Common preprocessing techniques include:
1. Removing null values
2. Encoding categorical variables
3. Feature scaling
4. Tokenization
5. Text normalization

Text normalization may include converting text to lowercase,
removing punctuation, and eliminating stopwords.

Preprocessing improves model performance and reduces computational overhead.

---

# SEMANTIC CHUNKING

Semantic chunking is the process of splitting large documents into
smaller meaningful sections while preserving contextual relationships.

Unlike fixed-size chunking, semantic chunking attempts to split text
at logical boundaries such as:
- Paragraphs
- Headings
- Sentences
- Topic changes

This approach improves retrieval quality in Retrieval-Augmented Generation (RAG) systems.

For example, a chunk discussing vector databases should not be merged
with unrelated information about image processing.

---

# VECTOR DATABASES

Vector databases store embeddings generated from machine learning models.

Embeddings are numerical vector representations of text, images, or audio.
These vectors help systems perform semantic similarity searches efficiently.

Popular vector databases include:
- Pinecone
- Weaviate
- FAISS
- ChromaDB

Vector search enables AI systems to retrieve contextually relevant information.

---

# EMBEDDING MODELS

Embedding models convert text into dense numerical vectors.

Popular embedding models include:
- OpenAI embeddings
- BGE embeddings
- Sentence Transformers
- E5 embeddings

Good embeddings capture semantic meaning effectively.

Embeddings are often used in:
- Search systems
- Recommendation engines
- Chatbots
- Document retrieval systems

---

# RETRIEVAL AUGMENTED GENERATION

Retrieval-Augmented Generation (RAG) combines retrieval systems
with large language models.

The workflow typically includes:
1. Chunking documents
2. Creating embeddings
3. Storing vectors
4. Retrieving relevant chunks
5. Generating answers

RAG systems improve factual accuracy by grounding responses in external knowledge.

---

# CONCLUSION

Semantic chunking is an essential technique in modern AI pipelines.
It improves retrieval quality, preserves contextual information,
and enhances the performance of language models.

Combining semantic chunking with vector databases and embedding models
creates powerful retrieval systems capable of handling large-scale documents.
"""

    return document





document=create_dummy_document()
chunked_document=semantic_chunking(document)
print("Total semantic chunks",len(chunked_document))
print("Chunk one characters",len(chunked_document[0].page_content))
print("Chunk one character section",chunked_document[0].metadata['section'])
print("Chunk one semantic density",chunked_document[0].metadata['semantic_density'])

