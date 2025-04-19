import requests
import os
from dotenv import load_dotenv

load_dotenv("db.env")

db_host = os.environ.get("DB_HOST")
db_port = os.environ.get("DB_PORT")

def retrive_relevant_context(text: str) :
    res = requests.get(f'http://{db_host}:{db_port}/query/{text}')

    if res.status_code == 200 :
        return res.content
    
    return "Error"