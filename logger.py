def log_message(message: str):
    with open("log.txt", 'a') as log_file:
        log_file.write(message + '\n')
    print(message)
