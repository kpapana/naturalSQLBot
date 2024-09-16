import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history =[]

st.set_page_config(page_title="Talk to your Data", page_icon="🤖")
st.title('Talk to your Data')

#get response

def get_response(user_query, chat_history):
    
    db_user=os.getenv("db_user")
    database_name=os.getenv("database_name")
    db_password=os.getenv("db_password")
    host_name=os.getenv("host_name")
    port=os.getenv("port")

    db_uri = "mysql+mysqlconnector://{db_user}:{db_password}@{host_name}:{port}/{database_name}".format(
        db_user=db_user,
        db_password=db_password,
        host_name=host_name,
        port=port,
        database_name=database_name
    )
    db=SQLDatabase.from_uri(db_uri)
    

    sql_chain_template = """Based on the table schema and chat history below, write a SQL query that would answer the user's question:
    {schema}

    Chat History: {chat_history}
    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_template(sql_chain_template)

    full_chain_template = """Based on the table schema, question, sql query, chat history and sql response below, write a natural language response:
    {schema}

    Chat History: {chat_history}
    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""
    prompt_response = ChatPromptTemplate.from_template(full_chain_template)
    
    
        
    llm=ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.1, openai_api_key=os.getenv("OPENAI_API_KEY"))

    sql_chain = (
        RunnablePassthrough.assign(
            schema=lambda schema:db.get_table_info(), 
            chat_history=lambda chat_history: chat_history
        )
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    full_chain = (
        RunnablePassthrough.assign(
            query=sql_chain
        ).assign(
            schema=lambda schema: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
            chat_history=lambda chat_history: chat_history
        )
        | prompt_response
        | llm
        | StrOutputParser()
    )

    # response = """
    # SQL Query: {generated_query} \n
    # AI response: {generated_response} \n
    # """
    # return response.format(
    #     generated_query=sql_chain.invoke({"question": user_query}),
    #     generated_response=full_chain.invoke({"question": user_query})
    # )

    return full_chain.invoke({"question": user_query})
    


#conversation
for message in st.session_state.chat_history:
    if isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)


# user input
user_query=st.chat_input("your message")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response =  get_response(user_query, st.session_state.chat_history)
        st.markdown(ai_response)
    st.session_state.chat_history.append(AIMessage(ai_response))