# Useful

- to kill process allocating some specific port
    - identify the process using `lsof -i -P`
    - to remove the process, simply killing its parent process should do the trick
        - 
