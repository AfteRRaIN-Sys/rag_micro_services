import os
import subprocess
from dotenv import load_dotenv

import fastapi
from fastapi import FastAPI
from langchain_ollama import OllamaLLM

import db_client

load_dotenv("service.env")

BACKEND_PORT = os.environ.get("PORT")

app = FastAPI()

MODEL_CKPT = "llama3.2"
model = OllamaLLM(model=MODEL_CKPT)


@app.get("/")
def test():
    return {"message": "hello backend!"}


@app.get("/qa/{question}")
def qa(question: str):

    print(f"qa : {question}")

    # retrive relevant text from vector db
    retrived_data = db_client.retrive_relevant_context(question)

    if retrived_data == "Error":
        return {"status_code": 500, "message": "Retrival Error!"}

    context = "\n".join([e[0] for e in retrived_data["documents"]])

    # generate context-augmented prompt
    augmented_prompt = f"""Context: {context}\n\nQuestion: {question}\nAnswer:"""

    # generate response
    response = model.invoke(augmented_prompt)

    return {"query": question, "context": context, "response": response}


def main():
    """
    2 functions
    - personalized ver
    - overall ver
    """

    # download data
    pass

    # encode all data
    pass

    # save to database
    pass

    # start service
    pass


if __name__ == "__main__":
    print("start app!")
    subprocess.run(f"uvicorn server:app --host 0.0.0.0 --port {BACKEND_PORT}".split())
