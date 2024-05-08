import datetime

log_messages = []

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] " + message
    log_messages.append(formatted_message)

def get_log_messages():
    return '\n'.join(log_messages)