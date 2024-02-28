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
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

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

class QuestionAnswer(BaseModel):
    Question: str
    SQL_Query: str
    Ans: str

# Config Path
config_path = "Ron_sql_db.ini"
current_directiry= os.getcwd()+'\\'+config_path


# Read from config file
get_section = configparser.ConfigParser(interpolation=None)
get_section.read(current_directiry)
max_token = get_section.get('Gpt_model_conf', 'max_token')
temperature = get_section.get('Gpt_model_conf', 'temperature')
PROMPT = get_section.get('PROMPT_CONFIG', 'default_system_prompt')


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
# memory = ConversationBufferWindowMemory(input_key='input', memory_key="history",k=3)



#API to get the DB credentials
@app.post("/connection_input/")
async def user_credentials(
    host: str,
    user: str,
    password: str,
    database: str,
    port: int,
    table_names: List[str]
):
    config['DATABASE'] = {
        'host': host,
        'user': user,
        'password': password,
        'database': database,
        'port': str(port),
        'table_names': ','.join(table_names)}

    with open('Ron_sql_db.ini', 'w') as configfile:
        config.write(configfile)
        
    return {"message": "Credential updated sucessfully!!!"}

user = get_section.get('DATABASE', 'user')
password = get_section.get('DATABASE', 'password')
db_name = get_section.get('DATABASE', 'database')
host = get_section.get('DATABASE', 'host')
port = get_section.get('DATABASE', 'port')



# Function to extracting the table schema information
def check_table_schema():
    # Read configuration from .ini file
    config = configparser.ConfigParser()
    config.read('Ron_sql_db.ini')
    user = get_section.get('DATABASE', 'user')
    password = get_section.get('DATABASE', 'password')
    db_name = get_section.get('DATABASE', 'database')
    host = get_section.get('DATABASE', 'host')
    port = get_section.get('DATABASE', 'port')
    table_names_str = config.get('DATABASE', 'table_names')
    # Split the comma-separated string into a list
    table_name_db = table_names_str.split(',')

    try:
        # Connect to MySQL database
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=db_name,
            port = port
        )
        
        # Check if connection is successful
        if conn.is_connected():
            logging.info("Connected to MySQL database on localhost")
            
            # Initialize an empty string to concatenate schema strings
            all_schema_strings = ""
            # Loop through each table name
            for table_name in table_name_db:
                # Query the table schema
                cursor = conn.cursor()
                cursor.execute(f"DESCRIBE {table_name}")  # Modified query to retrieve table schema
                schema_rows = cursor.fetchall()

                # Format the schema into a string for the current table
                schema_string = f"Schema for table '{table_name}':\nCREATE TABLE {table_name} (\n"
                for row in schema_rows:
                    column_name = row[0]  # Modified index to match MySQL DESCRIBE output
                    data_type = row[1]    # Modified index to match MySQL DESCRIBE output
                    schema_string += f'    "{column_name}" {data_type},\n'
                schema_string = schema_string.rstrip(',\n') + "\n)\n\n"

                # Concatenate schema string for the current table to the overall schema string
                all_schema_strings += schema_string

            return all_schema_strings
    
    except mysql.connector.Error as e:
        logging.error(f"Error connecting to MySQL database: {e}")
        print(f"Error connecting to MySQL database: {e}")
    except Exception as ex:
        logging.error(f"Error: {ex}")
        print(f"Error: {ex}")


def create_system_prompt(domain_info: str, schema_string: str, Question_Answer: str):
    """
    Create the System_Prompt string.

    Parameters:
    - domain_info (str): Information about the domain.
    - schema_string (str): Schema string for the table.
    - Question_Answer (str): Examples of questions and answers.

    Returns:
    - str: The constructed System_Prompt string.
    """
    formatted_text = Question_Answer

    formatted_text += "\n"  # Add a newline between each set of question, query, and answer
    try:
        System_Prompt = (
            domain_info + "\n\n" +
            "Use the following table schema:\n" +
            schema_string + "\n" +
            "Following question, answer examples:\n" +
            formatted_text + "\n" +
            "Question: {question}\n SQL Query: {query} \n SQL Result: {result} \n Answer:")
        
    except:
        schema_string= """CREATE TABLE product_frida (
        "Name" int,
        "Legacy SKU Name" text,
        "UPC Code" bigint,
        "Inner GTIN-14" bigint,
        "Master GTIN-14" bigint,s
        "Product Packaged Length" double,
        "Product Packaged Width" double,
        "Product Packaged Height" double,
        "Weight" double,
        "Weight Units" text,
        "MSRP (USD)" text,
        "Country of Origin" text,
        "Minimum Qty" int,
        "Wholesale Price" text,
        "Sales Description" text
    )


    CREATE TABLE ticket_details (
        "Ticket ID" int,
        "Ticket created - Date" text,
        "Secondary" text,
        "Requester email" text,
        "Resolution" text,
        "First Name" text,
        "Last Name" text,
        "Tickets" int
    )"""
        System_Prompt = (
        domain_info + "\n\n" +
        "Use the following table schema:\n" +
        schema_string + "\n" +
        "Following question, answer examples:\n" +
        formatted_text + "\n" +
        "Question: {question}\n SQL Query: {query} \n SQL Result: {result} \n Answer:")


    return System_Prompt

def convert_to_string(data):
    result = ""
    for item in data:
        result += f"Question: {item.Question}\n"
        result += f"SQL Query: {item.SQL_Query}\n"
        result += f"Answer: {item.Ans}\n\n"
    return result


#API to get the domain info and the sample QA for system prompt generation
@app.post("/prompt_input/")
async def upload_file(domain_info: str = "", question_answers: List[QuestionAnswer] = []):
    schema_string = check_table_schema()
    question_answers_prompt = convert_to_string(question_answers)
    system_prompt = create_system_prompt(domain_info,schema_string,question_answers_prompt)
#   Change the value of default_system_prompt
    config.set('PROMPT_CONFIG','default_system_prompt', system_prompt)

#    Save the changes back to the file
    with open('Ron_sql_db.ini', 'w') as configfile:
        config.write(configfile)
    if system_prompt:
         return {"message": "Table created successfully.", "system_prompt": system_prompt}
    else:
         return {"message": "Failed to create table."}



#Function to create the uri to feed into the LLM
def create_database_uri(username, password, hostname, database_name):
    # Explicit input for the password
    password_encoded = urllib.parse.quote_plus(password)
    
    # Create the database URI
    uri = f"mysql+pymysql://{username}:{password_encoded}@{hostname}/{database_name}"
    db = SQLDatabase.from_uri(uri)
    return db


#API to get the query and generate the LLM response
@app.post("/query")
async def run_query(question: str):
    db = create_database_uri(user,password,host,db_name)
    answer_prompt=PromptTemplate.from_template(PROMPT)
    try:
        
        write_query = create_sql_query_chain(llm, db)
        execute_query = QuerySQLDataBaseTool(db=db)
        answer = answer_prompt | llm | StrOutputParser()
        chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer
        )
 
        result = chain.invoke({"question": question})
        modified_string = result.replace("AI: ", "")
 
        return modified_string
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# API to get the configurable parameter of LLM
@app.post("/config_change")
async def run_query(ai_chat: ai_chatbot_param):  
    try:  

        #config_path = f'Ron_Config\\{ai_chat.client_name}_config.ini'
        current_directiry= os.getcwd()+'\\'+config_path
        print(config_path)
        config = configparser.ConfigParser()
        config.read(current_directiry)
        changes_ai_chatbot = {
            "Gpt_model_conf": {
                "temperature": ai_chat.temperature,
                "max_token": ai_chat.max_token,
            },
            # "PROMPT_CONFIG": {
            #     "default_system_prompt": ai_chat.system_prompt,
            # }
        }

        for section, section_val in changes_ai_chatbot.items():
            for key, value in section_val.items():
                if value is not None:
                    config.set(section, key, str(value))

        with open(config_path, 'w') as configfile:
            config.write(configfile)
            # with open(config_path, 'w') as configfile:
            #     config.write(configfile)
        return {"message": "succesfully config updated"}
    except Exception as e:
        return {"error": f"something went wrong {str(e)}"}


    

#Run the uvicorn app 
if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=8000)

