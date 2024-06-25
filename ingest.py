import langchain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

import os

from constants import CHROMA_SETTINGS

persist_directory='db'
def main():
    for root, dirs, files in os.walk('docs'):
        for file in files:
            if file.endswith('pdf'):
                loader=PDFMinerLoader(os.path.join(root,file))
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=500)
    text=text_splitter.split(documents)
    
    ##creating embeddings
    embeddings=SentenceTransformerEmbeddings(model_name='all_MiniLM-L6-v2')
    
    #creating vector store
    db=Chroma.from_documents(text,embeddings,persist_directory=persist_directory,
                             client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None


if __name__=="__main__":
    main()
    