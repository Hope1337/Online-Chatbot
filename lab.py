import json

with open("data/api_aut.json", "r") as f:
    data = json.load(f)

print(data)
