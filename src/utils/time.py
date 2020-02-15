from datetime import datetime

DATETIME_TEMPLATE = "%Y-%m-%d_%H-%M-%S.%f"

def get_timestamp():
    return datetime.now().strftime(DATETIME_TEMPLATE)