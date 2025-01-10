import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from add_document import initialize_vectorstore
from langchain.chains import RetrievalQA

load_dotenv()

st.title("TLVTTS Chat Bot")

def create_qa_chain():
    vectorstore = initialize_vectorstore()
    callback = StreamlitCallbackHandler(st.container())

    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_API_MODEL"),
        temperature=os.getenv("OPENAI_API_TEMPERATURE"),
        streaming=True,
        callbacks=[callback],
    )

    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What's up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
        
    # これまでの会話も含めた回答を行う
    conversation_history = "\n".join(
        [f"{message['role']}: {message['content']}" for message in st.session_state.messages]
    )
    
    with st.chat_message("assistant"):
        qa_chain = create_qa_chain()
        response = qa_chain.invoke(prompt)

    # Open AI API からの回答を履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": response["result"]})

    # アシスタントの応答を表示
    st.chat_message("assistant").markdown(response["result"])
