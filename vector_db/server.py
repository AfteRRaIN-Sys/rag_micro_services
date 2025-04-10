import fastapi
from fastapi import FastAPI

import app as retriever

app = FastAPI()

vector_db_client = retriever.start()


@app.get("/")
def read_root():
    return {"text": "hello world!"}


@app.get("/query/{text}")
def query_relevant_text(text: str):
    # vector_db
    return {"query": text, "result": vector_db_client.retrive_relevant_context(text)}


@app.get("/qa/{text}")
def query_relevant_text(text: str):
    # vector_db
    return {"question": text}
    # return {"query": text, "result": vector_db_client.retrive_relevant_context(text)}


# if __name__ == "__main__":
#     pass
