# Useful

- to kill process allocating some specific port
  - identify the process using `lsof -i -P`
  - to remove the process, simply killing its parent process should do the trick

# Service Structure

- frontend

  - port: 3000

- backend

  - port: 8000

- vectordb
  - cmd
    - `chroma run --path chroma/`
    - `uvicorn server:app --host 0.0.0.0 --port 8080`
    <!-- - `fastapi dev server.py` -->
  - port: 8000
