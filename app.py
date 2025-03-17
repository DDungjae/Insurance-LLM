import os
import streamlit as st
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from rag import embeddings_model, retriever_from_llm, vectorstore, prompt, format_docs
from google_drive_downloader import GoogleDriveDownloader as gdd
import faiss
import pickle

FILE_ID_CHILD = "1iWPengosCT11IUyyNrZyqTAb-x8k3KDP"
FILE_NAME_CHILD = "index_child.faiss"
FILE_ID_1 = "1-AW6F3lnGJjjy-l_lZLHvf3CyozL9VOi"
FILE_NAME_1 = "index_1.faiss"
FILE_ID_2 = "1Z37sYjlHJpz3JrEbkbjrCIJQfZi5vB_9"
FILE_NAME_2 = "index_2.faiss"

file_list = [[FILE_ID_CHILD, FILE_NAME_CHILD], [FILE_ID_1, FILE_NAME_1], [FILE_ID_2, FILE_NAME_2]]

def download_faiss_index(file_id, file_name):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        st.error("❌ 파일 다운로드 실패!")

# Streamlit 실행 시 Faiss DB 다운로드
for ids, names in file_list:
    if not os.path.exists(names):
        download_faiss_index(ids, names)

# Faiss 인덱스 로드
for ids, names in file_list:
    index = faiss.read_index(FILE_NAME)
    

st.title("Insurance LLM")
st.subheader("using RAG")

embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

vectorstore_contract1 = FAISS.load_local('./db_contract/faiss', embeddings_model, allow_dangerous_deserialization=True)
vectorstore_contract2 = FAISS.load_local('./db_contract2/faiss', embeddings_model, allow_dangerous_deserialization=True)
vectorstore_contract_child = FAISS.load_local('./db_child_contract/faiss', embeddings_model, allow_dangerous_deserialization=True)

retriever_1= vectorstore_contract1.as_retriever()
retriever_2= vectorstore_contract2.as_retriever()
retriever_child= vectorstore_contract_child.as_retriever()

# 버튼 생성

select = st.selectbox("참조할 약관을 선택하세요",["약관1", "약관2", "약관_자녀"])

if select == "약관1":
    retriever_from_llm = retriever_1
if select == "약관2":
    retriever_from_llm = retriever_2
if select == "약관_자녀":
    retriever_from_llm = retriever_child

template = '''
You should answer the question by finding the evindance from
 the following context. Follow given instructions
 1. Provide the page number of context
 2. Answer in Korean.

 context: {context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model="gpt-4o")

chain = (
    {'context': retriever_from_llm | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def chatbot(question):
    yield chain.invoke(question)

if "messages" not in st.session_state:
    st.session_state.messages = []

messages = st.session_state.messages
for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("질문"):
    with st.chat_message("user"):
        st.write(question)
        messages.append({"role": "user", "content": question})
    with st.chat_message("ai"):
        answer = st.write_stream(chatbot(question))
        messages.append({"role": "ai", "content": answer})
        
