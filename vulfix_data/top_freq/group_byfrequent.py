import json
import re
from collections import defaultdict

with open("../../source_data/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

top_10_cwes = {
    "CWE-119", "CWE-20", "CWE-120", "CWE-200", "CWE-269",
    "CWE-400", "CWE-404", "CWE-772", "CWE-787", "CWE-190"
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
    file_name = f"top/{cwe_id}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
