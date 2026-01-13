import os
import json
import requests

root_dir = "../2025"
output_file = "filtered_cve.txt"

GITHUB_TOKEN = "yours GITHUB_TOKEN"

headers = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {GITHUB_TOKEN}"
}

valid_extensions = {"c", "cpp", "cc", "h", "hpp"}

filtered_cves = []

def get_commit_file_extensions(commit_url, headers=None):
    """获取 GitHub 提交的文件扩展名"""
    try:
        parts = commit_url.split("/")
        repo = f"{parts[3]}/{parts[4]}"
        commit_sha = parts[-1]

        api_url = f"https://api.github.com/repos/{repo}/commits/{commit_sha}"
        response = requests.get(api_url, headers=headers)

        if response.status_code == 403:
            return set()
        elif response.status_code != 200:
            return set()

        commit_data = response.json()
        extensions = {file["filename"].split(".")[-1] for file in commit_data.get("files", []) if "." in file["filename"]}
        return extensions

    except Exception as e:
        return set()

for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            if filename.endswith(".json"):
                file_path = os.path.join(subdir_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if data.get("cveMetadata", {}).get("assignerShortName", "") != "GitHub_M":
                        continue

                    github_commit_urls = [
                        ref["url"] for ref in data.get("containers", {}).get("cna", {}).get("references", [])
                        if "github.com" in ref.get("url", "") and "/commit/" in ref["url"]
                    ]

                    cwe_id = "N/A"
                    problem_types = data.get("containers", {}).get("cna", {}).get("problemTypes", [])
                    for problem in problem_types:
                        for desc in problem.get("descriptions", []):
                            if desc.get("type") == "CWE":
                                cwe_id = desc.get("cweId", "N/A")
                                break
                        if cwe_id != "N/A":
                            break
                    for url in github_commit_urls:
                        extensions = get_commit_file_extensions(url, headers=headers)
                        if extensions & valid_extensions:
                            cve_id = data["cveMetadata"]["cveId"]
                            filtered_cves.append(f"{cve_id} {url} {cwe_id}")
                        else:
                            print("no C/C++ code")

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"{file_path}，{e}")


if filtered_cves:
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n".join(filtered_cves) + "\n")
else:
    print("no CVE")
