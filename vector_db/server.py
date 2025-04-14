import fastapi
from fastapi import FastAPI
from langchain_ollama import OllamaLLM

import app as retriever

app = FastAPI()

model_ckpt = "llama3.2"
model = OllamaLLM(model=model_ckpt)

vector_db_client = retriever.start()


@app.get("/")
def read_root():
    return {"text": "hello world!"}


@app.get("/query/{text}")
def query_relevant_text(text: str):
    # vector_db
    return {"query": text, "result": vector_db_client.retrive_relevant_context(text)}


@app.get("/qa/{question}")
def qa(question: str):
    # vector_db
    # return {"question": prompt}

    # retrive relevant text
    context = vector_db_client.retrive_relevant_context(question)

    # generate context-augmented prompt
    augmented_prompt = f"""Context: {context}\n\nQuestion: {question}\nAnswer:"""

    # generate response
    response = model.invoke(augmented_prompt)

    return {"query": question, "context": context, "response": response}


# if __name__ == "__main__":
#     pass
