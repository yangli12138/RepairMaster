import json

with open("../test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

short_code = []
long_code = []
extra_long_code = []

for item in data:
    code = item["question"]
    code_length = len(code)

    if code_length > 1024:
        extra_long_code.append(item)
    elif code_length > 512:
        long_code.append(item)
    else:
        short_code.append(item)

with open("test_short.json", "w", encoding="utf-8") as f:
    json.dump(short_code, f, indent=4, ensure_ascii=False)

with open("test_long.json", "w", encoding="utf-8") as f:
    json.dump(long_code, f, indent=4, ensure_ascii=False)

with open("test_extra_long.json", "w", encoding="utf-8") as f:
    json.dump(extra_long_code, f, indent=4, ensure_ascii=False)
