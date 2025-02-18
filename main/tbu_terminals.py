# config.py
import json

CONFIG_FILE = "global_state.json"

# Initialize the file if it doesn't exist
try:
    with open(CONFIG_FILE, "r") as f:
        global_state = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    global_state = {'current-state': {}}  # Default value
    with open(CONFIG_FILE, "w") as f:
        json.dump(global_state, f)

def update_variable(value):
    global_state["current-state"] = value
    with open(CONFIG_FILE, "w") as f:
        json.dump(global_state, f)

def get_variable():
    with open(CONFIG_FILE, "r") as f:
        global_state = json.load(f)
    return global_state["current-state"]
