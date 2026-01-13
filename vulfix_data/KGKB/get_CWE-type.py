import json
import re
from collections import defaultdict


def extract_cwe_types(json_data):
    cwe_set = set()

    for item in json_data:
        question = item.get("question", "").strip()
        match = re.match(r'^CWE-(\d+)', question)
        if match:
            cwe_set.add(f"CWE-{match.group(1)}")

    return sorted(cwe_set, key=lambda x: int(x[4:]))


with open('../test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    cwe_list = extract_cwe_types(data)

with open('cwe_types.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(cwe_list))
