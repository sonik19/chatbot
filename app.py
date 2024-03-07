import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from llama_cpp import Llama

st.set_page_config(layout="wide")

st.write("# VitamotusBot")
st.write('Dobryj Den, **Vadim Igorevich**!')

user_query = st.text_input("Ask a question")
selected_option = st.selectbox(
    "Select a model type",
    ("Public Data", "Vitamotus Data")
)

if st.button('Submit'):

    if selected_option == "Public Data":
        model = Llama(model_path="./mistral-7b-instruct-v0.2.Q2_K.gguf")

        output = model(
            f"<s>[INST] {user_query} [/INST]",
            max_tokens=256    
        )
        res = output['choices'][0]['text']

    elif selected_option == "Vitamotus Data":

        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
        vectordb = Chroma(persist_directory="./db", embedding_function=embeddings)

        docs = vectordb.similarity_search(user_query)

        context = docs[0].page_content

        template = f"""<s>[INST] Answer the question based only on the following context (your answer must be short and informative):
        {context}

        Question: {user_query} [/INST]
        """

        model = Llama(model_path="./mistral-7b-instruct-v0.2.Q2_K.gguf")

        output = model(
            template,
            max_tokens=256    
        )
        res = output['choices'][0]['text']

    st.write(res)