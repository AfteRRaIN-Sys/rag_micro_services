# Useful

- to kill process allocating some specific port
  - identify the process using `lsof -i -P`
  - to remove the process, simply killing its parent process should do the trick

- using fastapi, sending python object as a response directly is allowed (since fastapi did the `json` conversion behind the scene)

- Example QA
  
  - question: `Who is Brown and what kind of shirt should i recommend him based on his favorite color and personality`
  
  - output: ![alt text](image.png)

- for the vscode to recognize the `requirements` files, make sure to use `requirements` prefix

---
# Service Structure

- frontend

  - port: 3000

- ChromaDB server
  - port: 8000

- backend
  - port: 8081

- vectordb
  - cmd
    - `chroma run --path chroma/`
    - `uvicorn server:app --host 0.0.0.0 --port 8080`
    <!-- - `fastapi dev server.py` -->
  - port: 8080
