import os
from config.settings import LOG_FILE

def log_message(message: str):
    """Logs a message to console and file."""
    print(message)
    # Ensure directory exists if log file is in a subdir
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')