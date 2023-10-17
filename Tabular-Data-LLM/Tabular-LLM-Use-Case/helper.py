# File for Functions
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import matplotlib.pyplot as plt
import re
import transformers
import pandas as pd
import chromadb
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import CSVLoader, DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RouterChain, SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
import random
import streamlit as st

tokenused = 0
def gettokens():
    return tokenused

def addToken(amount):
    tokenused = tokenused+amount

def runQuery(db: Chroma,llm: ChatOpenAI,query: str) -> str:
    # return "Testing "+query
    prompt1 = """ The user will provide you with query or request to make on a CSV file, there is a limited set of functionality which the app can do
    your job is to determine which of the following tasks best fits the user's request, we have provided examples below

    1. Listing set amount of items with Condition example: "List me 10 tables under $100"
    2. Existance Verification, example: "Are there any toys with a shipping weight more than 20 lb"
    3. Basic Retrieval, example: "What is the price of a bottle opener"
    4. Ranking/Maximum/Minumum Questions, example: "What is the most expensive bottle opener"
    5. Other/Unkown, example: "What is the weather in San Francisco"

    reply with the number of the task you deem best fitting and only the number, nothing else
    For example, "1"

    This is the user query: "{input}"
    """

    first_prompt = ChatPromptTemplate.from_template(prompt1)

    myLlm = llm
    chain1 = LLMChain(
        llm=llm,
        prompt=first_prompt,
        verbose=True,
    )
    
    try:
        out = chain1.run(query)
    except:
        return "INVALID API KEY, please re-enter/copy and paste your key above"


    if int(out) == 5:
        return runIdk(db=db,llm=myLlm,query=query)
    if int(out) == 1:
        return listSetCon(db=db,llm=llm,query=query)
    if int(out) == 2:
        return existVer(db=db,llm=llm,query=query)
    if int(out) == 3:
        return basicRet(db=db,llm=llm,query=query)
    if int(out) == 4:
        return runRank(db=db,llm=llm,query=query)
    
    
    return out

def listSetCon(db: Chroma,llm: ChatOpenAI,query: str):
    listText = """
    The user is going to make a query on a CSV file that asks you to list out a specific amount of documents with certain conditions.
    Your job is to extract these three things from the user's query:
    1. The amount of documents they want to list - Quantity
    2. The keyword for the documents they want - Keyword
    3. The condition they want the documents listed to follow (This can be blank) - Condition

    Give your answer in this format:
    [Keyword,Quantity,Condition]

    Here are examples:
    Example 1:
    Q: "List me 10 Desks under $100"
    A: [desk,10,under $100]

    Example 2:
    Q: "Show 5 toys that weigh more than 20 lb"
    A: [toys,5,weigh more than 20 lb]

    Example 3:
    Q: "List two random headphones"
    A: [headphone,2,none]

    The Keyword and Quantity cannot be blank but the condition may be. If the condition is not there condition = "none".
    This is the user query: "{input}"

    """
    listTemplate = ChatPromptTemplate.from_template(listText)

    chain4 = LLMChain(
        llm=llm,
        prompt=listTemplate,
        verbose=True,
    )

    listrawout = chain4.run(query)
    print(listrawout)
    listrawout = listrawout.strip("[")
    listrawout = listrawout.strip("]")
    listlist = listrawout.split(",")
    print(listlist)
    listsearch = db._collection.get(
    where_document={
          "$contains": listlist[0]

        }
    )

    loopText = """
    We will be giving you a piece of context between the hyphens and then a condition that it should follow
    say YES if the context fits the condition and NO if not
    Contex:
    ----------------------
    {input}
    ----------------------

    The condition is: """+listlist[2]+"""
    """
    counter = 0
    retL = []
    if listlist[2] != "none":
        for i in listsearch["documents"]:
            loopchain = LLMChain(
                llm=llm,
                prompt=ChatPromptTemplate.from_template(loopText),
                verbose=True,
            )

            ou = loopchain.run(i)
            if ou == "YES":
                counter += 1
                print(counter)
                retL.append(i)

            if counter == int(listlist[1]):
                break
        print(counter)
        print(retL)
    else:
        for i in listsearch["documents"]:
            retL.append(i)
            if counter == int(listlist[1]):
                break

    if len(retL) > 0:
        return " \n".join(retL)
    else:
        return "Couldn't find any results for: "+listlist[0]+", that follow condition: "+listlist[2]




def runIdk(db: Chroma,llm: ChatOpenAI,query: str):
    idkText = """
    You are an AI assistant who just recieved a query you cannot perform. The query is made on a CSV file, respond formally and kindly with 
    a different place the user should look for an answer to their query as you can only get information on the CSV file.
    Make sure to say let me know if you need anything else related directly to the file after

    The user's query is: "{input}"
    """
    idk_prompt = ChatPromptTemplate.from_template(idkText)
    idkChain = LLMChain(
        llm=llm,
        prompt=idk_prompt,
        verbose=True
    )

    out = idkChain.run(query)
    return out


def existVer(db: Chroma,llm: ChatOpenAI,query: str):
    promptExistance = """
        The User is making a query on a CSV file to verify if there exists a certain item that fullfills a condition.
        Your job is to determine what item the user is looking for and what condition based on the query they provide
        You need to look for these two things in the query>
        1. Item Keyword, there will always be an item keyword this cannot be blank
        2. The Condition, if you cannot find a condition say "None"

        Example 1: "Is there an Android Tablet under $20"
        In this case the Keyword is "Android Tablets" and the Condition is "price less than $20"

        Example 2: "Is there any red solo cups?"
        In this example, the Keyword is "Red Solo Cups" and the Condition is "None" as there is no condition

        Give your answer like this:
        [Keyword,Condition]

        This is the user query: "{input}"
        """
    prompt_existance_temp = ChatPromptTemplate.from_template(promptExistance)
    chain2 = LLMChain(
        llm=llm,
        prompt=prompt_existance_temp,
        verbose=True,
    )
    chain2out = chain2.run(query)
    chain2out = chain2out.strip("[")
    chain2out = chain2out.strip("]")
    chain2outList = chain2out.split(",")
    searcheddocs = db._collection.get(
    where_document={
          "$contains": chain2outList[0]

        }
    )
    for i in searcheddocs["documents"]:
        existanceSearchPrompt = ChatPromptTemplate.from_template("""
        Answer "YES" if condition is true, and "NO" if not
        The context is ---------------------
        {input}
        -----------------------------
        The condition is: """+chain2outList[1]+"""
        """)
        loopchain = LLMChain(
            llm=llm,
            prompt=existanceSearchPrompt,
            verbose=True,
            )

        if loopchain.run(i) == "YES":
            return "Yes, it exists:"+ i



def basicRet(db: Chroma,llm: ChatOpenAI,query: str):

    basicText = """The user is making a query to a CSV file to find out information about a specific piece of context.
    The users query will have two things, the keyword or keywords and what attribute of the context they want.
    Your job is to extract the keyword and attribute from the user's query.
    Here are some example queries the user may or may not make, and what we are extracting:

    Example 1:
    Q: "What is the price of a baseball hat?"
    A: [Baseball Hat,price]

    Example 2:
    Q:"Can you tell me a desk's shipping weight?"
    A: [desk,shipping weight]

    Example 3:
    Q: "What is the size of a disney DVD"
    A: [disney dvd,size]

    This is the user query: "{input}"
    """
    basicTemplate = ChatPromptTemplate.from_template(basicText)
    basicChain = LLMChain(
        llm=llm,
        prompt=basicTemplate,
        verbose=True,
    )

    basicOut = basicChain.run(query)
    basicOut = basicOut.strip("[")
    basicOut = basicOut.strip("]")
    basicOutList = basicOut.split(",")
    print(basicOutList)

    res = db._collection.get(
    where_document={
          "$contains": basicOutList[0]
     }
    )
    if len(res["documents"]) == 0:
        return "Sorry, I couldn't find anything in the document with query: "+basicOutList[0]
    reslist = list(res["documents"])
    choiceRes = random.choice(reslist)
    textPrompt = """
        You are going to be given a piece of context as well as a attribute of that context
        Your job is to reply with the value of the attribute of the context provided.
        Contex:
        ----------------------
        """+choiceRes+"""
        ----------------------

        If you cannot find the attribute in the given context, say something similar to "Sorry I couldn't find that exact piece of information for the keyword"
        The attribute is: {input}
    """

    chain3 = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(textPrompt),
        verbose=True,
    )

    return chain3.run(basicOutList[1])




def runRank(db: Chroma,llm: ChatOpenAI,query: str):
    rankText = """
    The user is making a query to a CSV file asking to find extrema of certain data details
    (for example the largest or smallest value of certain attribute).
    Your job is to extract these three things from the user's query:
    1. The Keyword - the context we are looking for in the file
    2. The Attribute - the attribute of the context we are going to rank
    3. The Rank type - Most or least/lowest or highest
    Give your answer like in the examples below and exactly as such, do not include any other information:

    Example 1:
    Q: "What is the most epxensive shoe horn?"
    A: [shoe horn,price,highest]

    Example 2:
    Q: "What is the smallest table?"
    A: [table,size,lowest]

    Example 3:
    Q: "What is most heavy toy?"
    A: [toy,weight,highest]


    This is the user query:
    "{input}"

    Give your answers like above in that exact format. No other words needed.
    """
    rankTemplate = ChatPromptTemplate.from_template(rankText)

    chain5 = LLMChain(
        llm=llm,
        prompt=rankTemplate,
        verbose=True,
    )

    rankOut = chain5.run(query)
    rankOut = rankOut.strip("[")
    rankOut = rankOut.strip("]")
    rankList = rankOut.split(",")
    print(rankList)
    ranksearch = db._collection.get(
    where_document={
          "$contains": rankList[0]

        }
    )
    print(len(ranksearch["documents"]))
    if len(ranksearch["documents"]) == 0:
        return rankList[0]+", \n and unfortunatley I couldn't find anything from there"
    
    rankLoopText = """
        We will be giving you a piece of context between the hyphens and then a attribute that follows
        Your job is to pull out the specified attribute of the context and only the attribute, so do not include smybols or units etc in your answer.

        Contex:
        ----------------------
        {input}
        ----------------------

        The attribute is: """+rankList[1]+"""

        Make sure to not include $, lbs, or anything similar in the answer we just want a number.
        """
    rankLoopTemplate = ChatPromptTemplate.from_template(rankLoopText)

    chain6 = LLMChain(
        llm=llm,
        prompt=rankLoopTemplate,
        verbose=True,
    )

    # Create a Dict to sort
    rankingDict = {}

    #Loop through
    for i in ranksearch["documents"]:
        chout = chain6.run(i)
        if chout.replace(".", "").isnumeric():
            rankingDict[i]  = float(chout)
        else:
            rankingDict[i] = 0

    #Sort Dict
    print(rankingDict)
    print(sorted(rankingDict.values()))
    if rankList[2] == "lowest":
        rankingDict = sorted(rankingDict)
        res = list(rankingDict.keys())[0]
        return "Here is the value we found "+ str(res)
    elif rankList[2] == "highest":
        rankingDict = sorted(rankingDict, reverse=True)
        res = list(rankingDict.keys())[0]
        return "Here is the value we found "+ str(res)



