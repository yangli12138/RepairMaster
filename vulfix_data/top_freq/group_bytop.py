import json
import re
from collections import defaultdict

with open("../../source_data/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

top_10_cwes = {
    "CWE-787", "CWE-89", "CWE-352", "CWE-22", "CWE-125",
    "CWE-78", "CWE-416", "CWE-20", "CWE-77", "CWE-190"
}


cwe_groups = defaultdict(list)

for item in data:
    question = item["question"]

    match = re.match(r"(CWE-\d+)", question)

    if match:
        cwe_id = match.group(1)
        if cwe_id in top_10_cwes:
            cwe_groups[cwe_id].append(item)

for cwe_id, items in cwe_groups.items():
    file_name = f"{cwe_id}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)

