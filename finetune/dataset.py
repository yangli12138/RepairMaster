import torch
import codecs
import random
from transformers import RobertaTokenizer
from tree_sitter import Language, Parser
import warnings


warnings.simplefilter('ignore', FutureWarning)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, shuffle=False, load_range=None):
        self.data = []
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
        self.parser = Parser()
        self.parser.set_language(self.JAVA_LANGUAGE)

        fp = codecs.open(file_path, 'r', 'utf-8')
        for l in fp.readlines():
            l = eval(l)
            buggy_function_before = l['buggy function before']
            buggy_line = l['buggy line']
            buggy_function_after = l['buggy function after']
            fixed_line = l['fixed line'] + tokenizer.eos_token

            ast_sequence = self.get_ast_sequence(buggy_function_before, buggy_line, buggy_function_after)

            inputs = buggy_function_before + '</s>' + buggy_line + '</s>' + buggy_function_after + '</s>' + ast_sequence
            outputs = fixed_line

            inputs = tokenizer.encode(inputs, return_tensors='pt')
            outputs = tokenizer.encode(outputs, return_tensors='pt')

            if inputs.size(1) > max_length or outputs.size(1) > max_length:
                continue

            self.data.append({
                'input_ids': inputs,
                'labels': outputs,
                'attention_mask': torch.ones(inputs.size()).long()
            })

            if len(self.data) % 10000 == 0:
                print('finish loading:', len(self.data))

            if load_range is not None and len(self.data) == load_range[1]:
                break

        if shuffle:
            random.shuffle(self.data)

        print(file_path, 'total size:', len(self.data))

        if load_range is not None:
            self.data = self.data[load_range[0]:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_ast_sequence(self, function_before, buggy_line, function_after):
        code = function_before + buggy_line + function_after
        ast = self.generate_ast(code)

        ast_sequence = self.ast_to_string(ast)
        return ast_sequence

    def generate_ast(self, code):
        tree = self.parser.parse(code.encode('utf-8'))
        return tree.root_node

    def ast_to_string(self, root_node):
        ast_sequence = self.traverse_node(root_node)
        return " ".join(ast_sequence)

    def traverse_node(self, node):
        sequence = []
        sequence.append(f'({node.type}')
        for child in node.children:
            sequence.extend(self.traverse_node(child))
        sequence.append(')')
        return sequence


def custom_collate(batch):
    batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
    max_input_len = max([b['input_ids'].size(1) for b in batch])
    max_output_len = max([b['labels'].size(1) for b in batch])
    for b in batch:
        batch_data['input_ids'].append(
            torch.cat([b['input_ids'], torch.zeros(1, max_input_len - b['input_ids'].size(1)).long()], dim=1))
        batch_data['labels'].append(
            torch.cat([b['labels'], torch.zeros(1, max_output_len - b['labels'].size(1)).fill_(-100).long()], dim=1))
        batch_data['attention_mask'].append(
            torch.cat([b['attention_mask'], torch.zeros(1, max_input_len - b['attention_mask'].size(1))], dim=1))
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    return batch_data
