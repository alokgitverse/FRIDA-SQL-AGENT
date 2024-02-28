# Import all the libraries ############
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import pandas as pd
import sqlite3
from fastapi import FastAPI, HTTPException
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.sql_database import SQLDatabase
from langchain.chat_models.openai import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import os
from fastapi.middleware.cors import CORSMiddleware
import openai
import uvicorn
from langchain.prompts.prompt import PromptTemplate
from fastapi import FastAPI, HTTPException
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.sql_database import SQLDatabase
from langchain.chat_models.openai import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import os
from fastapi.middleware.cors import CORSMiddleware
import openai
import uvicorn
from langchain.prompts.prompt import PromptTemplate
import configparser
from pydantic import BaseModel
from typing import Optional
import configparser
import mysql.connector
import logging
from sqlalchemy import create_engine
import urllib.parse
from typing import List
import uvicorn
import json
from operator import itemgetter
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain.tools.sql_database.tool import QuerySQLDataBaseTool,QuerySQLCheckerTool

# Load the configuration file
config = configparser.ConfigParser()
config.read('Ron_sql_db.ini')


# Create instance for Fast API
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SchemaResponse(BaseModel):
    schema1: str

class OurBaseModel(BaseModel):
    class Config:
        from_attribute = True

class ai_chatbot_param(OurBaseModel):
  #client_name: str
  temperature: Optional[float] = None
  max_token: Optional[int] = None
  #system_prompt: Optional[str] = None

class question_ans_list(OurBaseModel):
    Question: Optional[str] = None
    SQLQuery: Optional[str] = None
    Ans: Optional[str] = None

class question_ans(OurBaseModel):
    domain_info: str = ""
    question_answer: List[question_ans_list]



# Config Path
config_path = "Ron_sql_db.ini"
current_directiry= os.getcwd()+'\\'+config_path


# Read from config file
get_section = configparser.ConfigParser(interpolation=None)
get_section.read(current_directiry)
max_token = get_section.get('Gpt_model_conf', 'max_token')
temperature = get_section.get('Gpt_model_conf', 'temperature')
PROMPT = get_section.get('PROMPT_CONFIG', 'default_system_prompt')
sql_agent_prompt = get_section.get('PROMPT_CONFIG','agent_system_prompt')



#DB Credential
user = get_section.get('DATABASE', 'user')
password = get_section.get('DATABASE', 'password')
db_name = get_section.get('DATABASE', 'database')
host = get_section.get('DATABASE', 'host')
port = get_section.get('DATABASE', 'port')
table_names_str = config.get('DATABASE', 'table_names')
# Split the comma-separated string into a list
table_name_db = table_names_str.split(',')



# Extract API key for OPENAI
openai_key = get_section.get('API_KEY', 'open_ai_key')
os.environ["OPENAI_API_KEY"] = openai_key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize the LLM and Buffer memory
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
 
#memory = ConversationBufferMemory(input_key='input', memory_key="history")
memory = ConversationBufferWindowMemory(input_key='input', memory_key="history",k=3)

uri = f"mysql+pymysql://{user}:{password}@{host}/{db_name}"
db = SQLDatabase.from_uri(uri)

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
context = toolkit.get_context()
tools = toolkit.get_tools()