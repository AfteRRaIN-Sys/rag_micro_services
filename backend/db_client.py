import requests
import os
from dotenv import load_dotenv

import ast
import json

load_dotenv("db.env")

db_host = os.environ.get("DB_HOST")
db_port = os.environ.get("DB_PORT")


def retrive_relevant_context(text: str):

    # import time
    # time.sleep(1)

    url = f"http://{db_host}:{db_port}/query/{text}"
    res = requests.get(url)

    if res.status_code == 200:
        data = json.loads(res.content.decode("UTF-8"))["result"]
        return data

    return "Error"
