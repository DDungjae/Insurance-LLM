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
        