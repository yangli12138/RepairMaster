import json
import torch
from torch.utils.data import Dataset
from tree_sitter import Language, Parser
import warnings

warnings.simplefilter('ignore', FutureWarning)
JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)


def extract_structure_info(code_str):
    try:
        tree = parser.parse(bytes(code_str, "utf8"))
        root_node = tree.root_node

        nodes = []
        def traverse(node, depth=0):
            if node.type not in ("comment", "string", "identifier"):
                nodes.append('  ' * depth + node.type)
            for child in node.children:
                traverse(child, depth + 1)

        traverse(root_node)
        ast_text = "\n".join(nodes)
        return ast_text
    except Exception as e:
        print(f"[Warning] Structure extraction failed: {e}")
        return "NO_AST_INFO"

class Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, shuffle=False, load_range=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle = shuffle

        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line.strip()) for line in f]

        if load_range:
            self.data = self.data[load_range[0]:load_range[1]]

        if self.shuffle:
            from random import shuffle
            shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        buggy_code = item['buggy line'] + item['buggy function before'] + item['buggy function after']
        fixed_code = item['fixed line']
        struct_info = extract_structure_info(buggy_code)
        input_text = buggy_code + ' <struct> ' + struct_info

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            fixed_code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': model_inputs.input_ids.squeeze(0),
            'attention_mask': model_inputs.attention_mask.squeeze(0),
            'labels': labels.input_ids.squeeze(0)
        }

def custom_collate(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
