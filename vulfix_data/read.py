import pandas as pd
import json


with open('ture/bug_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(type(data), len(data))

