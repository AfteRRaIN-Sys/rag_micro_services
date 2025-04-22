import os
from dotenv import load_dotenv

import json

from fastapi import FastAPI
from langchain_ollama import OllamaLLM

import engine


load_dotenv("server.env")
PORT = os.environ.get("PORT")

app = FastAPI()

vector_db_client = engine.start()


@app.get("/")
def read_root():
    res = {"status": 200, "text": "hello vector db!"}
    return res


@app.get("/query/{text}")
def query_relevant_text(text: str):
    # vector_db
    res = {
        "status": 200,
        "query": text,
        "result": vector_db_client.retrive_relevant_context(text),
    }
    return res


@app.get("/query/{text}/{k}")
def query_relevant_text(text: str, k: int):
    # vector_db
    res = {
        "status": 200,
        "query": text,
        "result": vector_db_client.retrive_relevant_context(text, k),
    }
    return res


# @app.get("/qa/{question}")
# def qa(question: str):
#     # vector_db
#     # return {"question": prompt}

#     # retrive relevant text
#     context = vector_db_client.retrive_relevant_context(question)

#     # generate context-augmented prompt
#     augmented_prompt = f"""Context: {context}\n\nQuestion: {question}\nAnswer:"""

#     # generate response
#     response = model.invoke(augmented_prompt)

#     return {"query": question, "context": context, "response": response}


if __name__ == "__main__":

    import subprocess

    subprocess.run(f"uvicorn server:app --host 0.0.0.0 --port {PORT}".split())
