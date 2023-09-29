# file to hold UI and run streamlit

# To do
# We need an api key input
# Show token used so far
# Try and make colors of Critical River more apparent in theme
# Find the other CSV sheet and break it into very small piece(under 100 rows) so that it can be processed quickly
import numpy as np
import random
from langchain.chat_models import ChatOpenAI
import matplotlib.pyplot as plt
import pandas as pd
import chromadb
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import CSVLoader, DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions
import streamlit as st
from PIL import Image
from langchain.callbacks import get_openai_callback
import sys
sys.path.insert(0,"/Desktop/summerfinapp")
from helper import runQuery
st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.sidebar.header("📊 CSV QA")
image = Image.open('criticalriverimage.png')
st.image(image)

embedding_function = HuggingFaceEmbeddings(model_name="deepset/all-mpnet-base-v2-table")
with st.spinner('Loading'):
    db = Chroma(persist_directory="./mychroma2/", embedding_function=embedding_function)

number_rows = db._collection.count()

st.write("Your Vectorized CSV file has "+str(number_rows)+" entries.")
OPENAI_API_TOKEN = st.text_input("Your Open AI API Key","TYPE KEY HERE AND PRESS ENTER TO APPLY")
# OPENAI_API_TOKEN = "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"
llm  = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    openai_api_key=OPENAI_API_TOKEN,
    temperature=0.0
)
st.title("DataChatr")
st.write("Powered by GPT 3.5")



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your CSV assistant, ask me questions about your CSV files(I have been fine tuned on Product data so I do retail queries very well!)"}]
    st.session_state.messages.append({"role": "assistant", "content": """
                                      Ask me things like:\n

                                      Show me 5 of instances of X is in the document

                                      What is the (attribute) of (item) in CSV"""})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask away!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": "Thinking...💭"})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    st.markdown("Thinking...💭")
    with st.chat_message("assistant"):
        # message_placeholder = st.empty()
        full_response = runQuery(db,llm,prompt)
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
