# File for Loading CSV
import streamlit as st
from langchain.document_loaders import CSVLoader, DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import sys

print("start")
st.set_page_config(
    page_title="Vectorize CSV",
    page_icon="👋",
)
st.sidebar.success("Vectorize Your CSV, then QA it")

def setupDB(filename):
    dataframe = pd.read_csv(filename)
    st.caption("Preview of file you uploaded:")
    st.write(dataframe)
    dataframe.to_csv('userdata.csv')
    loader = CSVLoader('userdata.csv')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    idList = [str(i) for i in range(1, len(docs) + 1)]
    print(len(docs))
    embedding_function = HuggingFaceEmbeddings(model_name="deepset/all-mpnet-base-v2-table")
    print("Starting DB")
    db = Chroma.from_documents(docs, embedding_function, ids=idList, persist_directory="./mychroma")
    st.success('Document is ready to be queried', icon="✅")
    print("Done")


st.title("Vectorize Your CSV here:")
st.write("Setup your CSV File for LLM Powered Question Answering")



filedata = st.file_uploader(label="Upload CSV file here",type="csv")

if st.button("Analyze File", type="secondary"):
    setupDB(filedata)
