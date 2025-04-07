import fastapi
from fastapi import FastAPI

import app as vector_db

app = FastAPI()
# vector_db = Vec


@app.get("/")
def read_root():
    return {"text": "hello world!"}


@app.get("/query/{text}")
def query_relevant_text(query: str):
    # vector_db.
    pass
