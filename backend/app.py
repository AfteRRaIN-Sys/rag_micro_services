import fastapi
from fastapi import FastAPI
from langchain_ollama import OllamaLLM

import client 

app = FastAPI()

model_ckpt = "llama3.2"
model = OllamaLLM(model=model_ckpt)


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
