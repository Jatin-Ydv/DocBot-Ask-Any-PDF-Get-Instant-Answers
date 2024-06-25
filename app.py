import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
import os 

checkpoint='LaMini-T5-738M'
offload_folder=os.path.abspath('C:\\Users\\DELL\\Desktop\\Projects\\Arena\\off_folder')
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
base_model=AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map='auto',
    offload_folder=offload_folder,
    torch_dtype=torch.float32
)

def llm_pipeline():
    pipe=pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm=HuggingFacePipeline(pipeline=pipe)
    return local_llm


@st.cache_resource
def qa_llm():
    llm=llm_pipeline()
    embeddings=SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    db=Chroma(persist_directory='db',embedding_function=embeddings,
              client_settings=CHROMA_SETTINGS)
    retriever=db.as_retriever()
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_input(instruction):
    response=''
    instruction=instruction
    qa=qa_llm()
    generated_text=qa(instruction)
    answer=generated_text['result']
    return answer,generated_text


def main():
    st.title('Search your PDF 🦜📄')
    with st.expander('About the App'):
        st.markdown(
            '***'
            'This is a generative AI powered question and answering app that answers the questions'
             'about the PDF file given.'
            ' ***'
        )
    Question=st.text_area('Enter your question')
    if st.button('Search'):
        st.info('Your question :'+Question)
        st.info(' The Answer')
        answer,metadata=process_input(Question)
        st.write(answer)
        st.write(metadata)
        
        
if __name__ =='__main__':
    main()