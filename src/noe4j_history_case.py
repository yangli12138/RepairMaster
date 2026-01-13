from py2neo import Graph, Node, Relationship
from datetime import datetime
import json
from transformers import RobertaTokenizer, RobertaModel, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import re
import faiss
import numpy as np


graph = Graph('bolt://localhost:xxxx', auth=("xxx", "xxx"), name='xxx')


def read_cwe_from_file(file_path):
    with open(file_path, 'r') as file:
        cwe_list = [line.strip() for line in file if line.strip()]
    return cwe_list


def add_cwe_to_graph(cwe_list):
    for cwe_id in cwe_list:
        existing = graph.nodes.match("CWE-TYPE", name=cwe_id).first()
        if not existing:
            cwe_node = Node("CWE-TYPE", name=f"{cwe_id}")
            graph.create(cwe_node)
            print(f"Added CWE-{cwe_id} to the graph")
        else:
            print(f"CWE-{cwe_id} already exists in the graph")


def add_patch_to_graph(patch_data):
    cve_id = patch_data["cve_num"]
    vulnerable_code = patch_data["func_before"]
    patch_code = patch_data["fix_line"]
    cwe_list = [patch_data["CWE_type"]]
    patch_node = Node("Patch", cve_id=cve_id, vulnerable_code=vulnerable_code, patch_code=patch_code)
    graph.create(patch_node)
    print(f"Added patch for CVE-{cve_id} to the graph")
    print(cwe_list)
    for cwe_id in cwe_list:
        cwe_node = graph.nodes.match("CWE-TYPE", name=cwe_id).first()
        if cwe_node:
            fix_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fix_relationship = Relationship(patch_node, "FIXES", cwe_node, fix_time=fix_time)
            graph.create(fix_relationship)
            print(f"Created 'FIXES' relationship between patch and {cwe_id}")

def add_description_to_graph(patch_data):
    cve_id = patch_data["cve_num"]
    cwe_id = patch_data["CWE_type"]
    commit_msg = patch_data.get("commit_msg", "")

    if not commit_msg.strip():
        print(f"No commit message found for {cve_id}, skipping.")
        return

    desc_node = Node("Description", text=commit_msg)
    graph.create(desc_node)
    print(f"Added Description node for {cve_id}")

    patch_node = graph.nodes.match("Patch", cve_id=cve_id).first()
    if patch_node:
        describe_fix_rel = Relationship(desc_node, "description fix", patch_node)
        graph.create(describe_fix_rel)
        print(f"Created 'description fix' relationship between Description and Patch of {cve_id}")

def add_repair_result_to_graph(patch_data):
    cve_id = patch_data["cve_num"]

    repair_node = Node("RepairResult", results="true")
    graph.create(repair_node)
    print(f"Added RepairResult node for {cve_id}")

    patch_node = graph.nodes.match("Patch", cve_id=cve_id).first()
    if patch_node:
        verify_rel = Relationship(patch_node, "verification results", repair_node)
        graph.create(verify_rel)
        print(f"Created 'verification results' relationship between Patch and RepairResult for {cve_id}")
    else:
        print(f"Patch node for {cve_id} not found, skipping repair result link.")

def add_case_to_graph(patch_data, case_id):
    cve_id = patch_data["cve_num"]

    case_node = Node("Case", id=case_id)
    graph.create(case_node)
    print(f"Created Case node #{case_id} for {cve_id}")

    desc_node = graph.nodes.match("Description", text=patch_data.get("commit_msg", "")).first()
    if desc_node:
        rel_desc = Relationship(desc_node, "constitute", case_node)
        graph.create(rel_desc)
        print(f"Linked Description ➝ Case #{case_id}")

    patch_node = graph.nodes.match("Patch", cve_id=cve_id).first()
    if patch_node:
        rel_patch = Relationship(patch_node, "constitute", case_node)
        graph.create(rel_patch)
        print(f"Linked Patch ➝ Case #{case_id}")

    repair_node = graph.nodes.match("RepairResult", results="true").first()
    if repair_node:
        rel_repair = Relationship(repair_node, "constitute", case_node)
        graph.create(rel_repair)
        print(f"Linked RepairResult ➝ Case #{case_id}")

def get_patches_by_cwe(cwe_id):
    name = cwe_id
    query = """
    match (n:`CWE-TYPE`{name:$cwe_id})-[FIXES]-(m) 
    return m
    """
    result = graph.run(query, cwe_id=cwe_id).data()
    return result

def get_codebert_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding

def extract_cwe_id(question):
    match = re.search(r'CWE-(\d+)', question)
    if match:
        return match.group(0)
    return None
# 主程序
if __name__ == "__main__":
    # cwe_file_path = "../vulfix_data/cwe_list_unique_sorted.txt"
    # cwe_list = read_cwe_from_file(cwe_file_path)
    # add_cwe_to_graph(cwe_list)
    #
    # json_file_path = "../vulfix_data/megavul_flitered.json"
    # with open(json_file_path, 'r') as file:
    #     json_data = json.load(file)
    #
    # for patch_data in json_data:
    #     add_patch_to_graph(patch_data)
    #
    # json_file_path = "../vulfix_data/megavul_flitered.json"
    # with open(json_file_path, 'r') as file:
    #     json_data = json.load(file)
    #
    # for patch_data in json_data:
    #     add_description_to_graph(patch_data)
    #
    # json_file_path = "../vulfix_data/megavul_flitered.json"
    # with open(json_file_path, 'r') as file:
    #     json_data = json.load(file)
    #
    # for patch_data in json_data:
    #     add_repair_result_to_graph(patch_data)
    #
    # json_file_path = "../vulfix_data/megavul_flitered.json"
    # with open(json_file_path, 'r') as file:
    #     json_data = json.load(file)
    #
    # case_counter = 1
    # for patch_data in json_data:
    #     add_case_to_graph(patch_data, case_counter)
    #     case_counter += 1

    model = RobertaModel.from_pretrained("../CodeBert")
    tokenizer = RobertaTokenizer.from_pretrained(".././CodeBert")

    with open('../vulfix_data/ture/bug_data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    cwe_ids = set()
    for entry in data:
        question = entry["question"]
        cwe_id = extract_cwe_id(question)
        cwe_ids.add(cwe_id)

    print(f"Unique CWE types to process: {len(cwe_ids)}")

    for cwe_id in cwe_ids:
        print(f"Processing {cwe_id}...")

        patches = get_patches_by_cwe(cwe_id)
        print(f"Found {len(patches)} patches for {cwe_id}")

        if len(patches) == 0:
            for entry in data:
                question = entry["question"]
                entry_cwe_id = extract_cwe_id(question)
                if entry_cwe_id == cwe_id:
                    ctx_entry = {
                        "id": "50000",
                        "title": f"{cwe_id} Historical Repair Cases Input Vulnerable Code Is: {cwe_id} ",
                        "text": f"{cwe_id} Fixed Code Lines are: ",
                    }
                    entry['ctxs'].insert(0, ctx_entry)

            with open('../vulfix_data/ture/bug_data.json', 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            continue

        patch_embeddings = []
        for patch in patches:
            patch_node = patch['m']
            vulnerable_code = patch_node.get('vulnerable_code', 'No vulnerable code')
            if vulnerable_code:
                patch_embedding = get_codebert_embedding(vulnerable_code)
                patch_embeddings.append(patch_embedding.squeeze().numpy())

        patch_embeddings_np = np.array(patch_embeddings)
        print(f"Shape of patch_embeddings_np: {patch_embeddings_np.shape}")

        for entry in data:
            question = entry["question"]
            entry_cwe_id = extract_cwe_id(question)
            if entry_cwe_id != cwe_id:
                continue

            question_embedding = get_codebert_embedding(question)
            question_embedding_np = question_embedding.numpy().reshape(1, -1)

            index = faiss.IndexFlatL2(patch_embeddings_np.shape[1])
            index.add(patch_embeddings_np)

            D, I = index.search(question_embedding_np, k=1)

            most_similar_patch_idx = I[0][0]
            most_similar_patch = patches[most_similar_patch_idx]['m']
            vulnerable_code = most_similar_patch.get('vulnerable_code', 'No vulnerable code')
            patch_code = most_similar_patch.get('patch_code', 'No patch code')

            ctx_entry = {
                "id": "50000",
                "title": vulnerable_code,
                "text": patch_code
            }

            entry['ctxs'].insert(0, ctx_entry)

        with open('../vulfix_data/ture/bug_data.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"Completed processing {cwe_id}.")
