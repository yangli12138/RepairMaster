import json
import re

def read_cwe_from_file(file_path):
    with open(file_path, 'r') as file:
        cwe_list = [line.strip() for line in file if line.strip()]
    return cwe_list

def filter_json_data(json_data, cwe_list):
    filtered_data = []
    for entry in json_data:
        cwe_ids = entry.get("cwe_ids", [])
        is_vul = entry.get("is_vul", False)

        if is_vul and any(cwe in cwe_list for cwe in cwe_ids):
            filtered_entry = {
                "cve_id": entry.get("cve_id", ""),
                "cwe_ids": entry.get("cwe_ids", []),
                "publish_date": entry.get("publish_date", ""),
                "commit_msg": entry.get("commit_msg", ""),
                "func_before": entry.get("func_before", ""),
                "diff_line_info": entry.get("diff_line_info", {})
            }
            filtered_data.append(filtered_entry)
    return filtered_data

def clean_function_code(function_code):
    function_code = re.sub(r'/\*.*?\*/', '', function_code, flags=re.DOTALL)
    function_code = re.sub(r'//.*$', '', function_code, flags=re.MULTILINE)
    function_code = '\n'.join([line.lstrip() for line in function_code.splitlines()])
    return function_code


def clean_diff_lines(diff_lines):
    cleaned_lines = []
    for line in diff_lines:
        cleaned_line = re.sub(r'/\*.*?\*/', '', line)
        cleaned_line = cleaned_line.lstrip()
        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line)
    return cleaned_lines

def process_json_data(json_data):
    processed_data = []

    for entry in json_data:
        if "func_before" in entry:
            entry["func_before"] = clean_function_code(entry["func_before"])

        if "diff_line_info" in entry:
            if "added_lines" in entry["diff_line_info"]:
                entry["diff_line_info"]["added_lines"] = clean_diff_lines(entry["diff_line_info"]["added_lines"])
            if "deleted_lines" in entry["diff_line_info"]:
                entry["diff_line_info"]["deleted_lines"] = clean_diff_lines(entry["diff_line_info"]["deleted_lines"])

            if not entry["diff_line_info"]["added_lines"]:
                continue
            if not entry["diff_line_info"]["deleted_lines"]:
                continue
        processed_data.append(entry)

    return processed_data
def process_json_data_sub(json_data):
    processed_data = []

    for entry in json_data:
        cwe_ids = entry.get("cwe_ids", [])
        cve_id = entry.get("cve_id", "")
        publish_date = entry.get("publish_date", "")
        commit_msg = entry.get("commit_msg", "")
        func_before = entry.get("func_before", "")
        diff_line_info = entry.get("diff_line_info", {})
        deleted_lines = diff_line_info.get("deleted_lines", [])
        added_lines = diff_line_info.get("added_lines", [])

        for cwe_id in cwe_ids:
            cwe_type = cwe_id
            update_time = publish_date.split('T')[0]

            func_before_with_cwe = f"{cwe_type} Historical Repair Cases Input Vulnerable Code Is: {cwe_type} {func_before}"

            for line in deleted_lines:
                func_before_with_cwe = func_before_with_cwe.replace(line, f"<vul-start>{line}<vul-end>")

            if added_lines:
                fix_line_parts = " <vul-start>".join(added_lines)
                fix_line = f"{cwe_type} Fixed Code Lines are: <vul-start>{fix_line_parts}<vul-end>"
            else:
                fix_line = f"{cwe_type} Fixed Code Lines are: "

            new_entry = {
                "CWE_type": cwe_type,
                "cve_num": cve_id,
                "update_time": update_time,
                "commit_msg": commit_msg,
                "func_before": func_before_with_cwe,
                "fix_line": fix_line
            }

            processed_data.append(new_entry)

    return processed_data
if __name__ == "__main__":
    cwe_file_path = "cwe_types.txt"
    cwe_list = read_cwe_from_file(cwe_file_path)

    json_file_path = "megavul.json"
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    filtered_data = filter_json_data(json_data, cwe_list)
    processed_data = process_json_data(filtered_data)
    processed_sub_data = process_json_data_sub(processed_data)
    print(f"Processed {len(processed_sub_data)} entries after filtering.")

    output_file_path = "megavul_flitered.json"
    with open(output_file_path, 'w') as output_file:
        json.dump(processed_sub_data, output_file, indent=4)
