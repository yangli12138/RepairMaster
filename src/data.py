import torch
import random
import json
import regex
import string
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='',
                 title_prefix='',
                 passage_prefix=''):
        self.data = data
        self.n_context = n_context
        self.sort_data()

    # 返回数据集中的样本数量
    def __len__(self):
        return len(self.data)

    # 返回数据集中的样本数量
    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target
        elif 'answers' in example:
            return random.choice(example['answers'])
        else:
            return None

    # 检查一个答案是否存在于给定的段落中
    def check_answers(self, answers, passage):
        def remove_articles(text):
            return regex.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        passage = white_space_fix(remove_articles(remove_punc(lower(passage.strip()))))
        for a in answers:
            a_new = white_space_fix(remove_articles(remove_punc(lower(a.strip()))))
            if a_new in passage:
                return 1
        return 0

    # 用于处理特定类型的回答匹配，特别是与CWE相关的匹配逻辑
    def check_answers_for_cwe(self, answers, passage):
        def remove_articles(text):
            return regex.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        # passage = white_space_fix(remove_articles(remove_punc(lower(passage.strip()))))
        cwe_type_from_passage = passage.split()[0]
        cwe_type_from_answer = answers[0].split()[0]

        if cwe_type_from_answer.lower().strip().replace('-', '')  ==  cwe_type_from_passage.lower().strip().replace('-', ''):
            return 1
        else:
            return 0

    def check_answers_for_cwe_all_pos (self, answers, passage):
        return 1

    def check_answers_for_cwe_trees(self, answers, passage):

        # 用于解析从cwe_tree.txt文件中读取的每一行数据
        def line2key_values(string_):
            d = string_
            key_ = d.split(':')[0].strip()[1:-1]
            values = d.split(':')[1].strip()[1:-1].strip().split(',')
            values = [v.strip()[1:-1] for v in values]
            new_values = []
            for v in values:
                if len(v) > 0:
                    new_values.append(v)
            values = new_values

            return key_, values

        # 用于查找一个节点（cur_node）在给定的父子关系字典中的父节点、子节点以及同层的其他兄弟节点
        def find_near_child_and_parent_and_peer(cur_node, parent_child_dict):
            near_child, near_parent, near_peer = [], [], []
            if cur_node in parent_child_dict.keys():
                near_child = parent_child_dict[cur_node]
            for key_ in parent_child_dict.keys():
                if (key_ != cur_node) and (cur_node in parent_child_dict[key_]):
                    near_parent = key_
                    near_peer = parent_child_dict[key_]
                    near_peer.remove(cur_node)
            return near_parent, near_child, near_peer

        with open('cwe_tree.txt') as f:
            cwe_tree_relation_data = f.readlines()
        parent_child_dict = dict()
        for i in range(len(cwe_tree_relation_data)):
            parent, childs = line2key_values(cwe_tree_relation_data[i].strip()[1:-1])
            parent_child_dict[parent] = childs

        def remove_articles(text):
            return regex.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()


        cwe_type_from_passage = passage.split()[0]
        cwe_type_from_answer = answers[0].split()[0]

        near_parent, near_child, near_peer = find_near_child_and_parent_and_peer(cwe_type_from_answer.strip(), parent_child_dict)

        if cwe_type_from_answer.lower().strip().replace('-', '')  ==  cwe_type_from_passage.lower().strip().replace('-', ''):
            return 1
        # elif (cwe_type_from_passage.strip() in near_parent) or (cwe_type_from_passage.strip() in near_child) or (cwe_type_from_passage.strip() in near_peer):
        #    print('near child, parent, peer')
        #    return 1
        else:
            return 0

    # 返回一个样本的数据，包括问题、目标答案、候选答案（修复后的代码）、上下文（如相关段落），以及一个"golden"标签，表示该段落是否与输入的CWE类型一致
    def __getitem__(self, index):
        example = self.data[index]
        question = example['question']
        target = self.get_target(example)
        answers = example['answers']

        if 'ctxs' in example and self.n_context is not None:
            contexts = example['ctxs'][:self.n_context]
            passages = [c['title']+' ' +c['text'] for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            golden = [1] + [self.check_answers_for_cwe_trees(answers, c['title'] +'. ' + c['text']) for c in contexts]
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores, golden = None, None, None

        return {
            'index' : index, ##no use
            'question' : question, ## vulnerable input code (first 512 tokens)
            'target' : target, ## the fixed code
            'candidate_answers': answers, # all possible fixed code (as we only have a single fixed code by developers, thus, this list only have a single element)
            'passages' : passages, # the expanded snippets: vulnerable input code (after first 512 tokens), AST, CWE examples, CWE name and descriptions...
            'scores' : scores, ##no use
            'golden' : golden # whether a snippet in the "passages" refers to the same CWE type of the vulnerable input code or not
        }

    # 如果上下文包含score字段，则该方法会按socre的分数对上下文进行排序
    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    # 该方法用于返回指定索引的原始数据
    def get_example(self, index):
        return self.data[index]

# 用于将一批文本段落（batch_text_passages）转换为模型可接受的输入格式
def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

# 将文本段落（batch_text_passages）和答案（batch_answers）一起编码成模型可接受的输入格式，并为每个答案标记其在文本中的 token 范围
def encode_passages_spans(batch_text_passages, batch_answers, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    answers_token_ids = []
    for k, text_passages in enumerate(batch_text_passages):
        answer = batch_answers[k].lower().strip()
        answer_starts_lens = [(psg.lower().index(answer), len(answer)) if answer in psg.lower() and psg.lower().index(answer)+ len(answer) <max_length else None for i, psg in enumerate(text_passages)]
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True,
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

        offset_mapping = p['offset_mapping']
        answer_token_id = []
        for i, a_s_l in enumerate(answer_starts_lens):
            o_m = offset_mapping[i]
            answer_token_start_end = []
            if a_s_l is not None:
                a_s = a_s_l[0]
                a_e = a_s_l[0] + a_s_l[1]
                for j, map in enumerate(o_m):
                    if a_s >= map[0] and a_s < map[1]:
                        if len(answer_token_start_end) == 0:
                            answer_token_start_end.append(j)
                    if a_e > map[0] and a_e <= map[1]:
                        assert len(answer_token_start_end) == 1
                        answer_token_start_end.append(j)
                        break
            else:
                answer_token_start_end = [max_length, max_length]
            if len(answer_token_start_end) < 2:
                print("*******")
                answer_token_start_end = [max_length, max_length]
            answer_token_id.append(answer_token_start_end)
        answers_token_ids.append(answer_token_id)
    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool(), torch.LongTensor(answers_token_ids)

# 目的是在处理一批文本段落时，标记出文本中哪些 tokens 对应于候选答案的位置
def encode_passages_group_tagger(batch_text_passages, batch_candidate_answers, tokenizer, max_length):
    psg_num = len(batch_text_passages[0])
    passage_ids, passage_masks, token_labels = [], [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True,
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

        answers = [ans.lower().strip() for ans in batch_candidate_answers[k]]
        passages = [psg.lower().strip() for psg in batch_text_passages[k]]
        ans_loc = [[] for _ in range(psg_num)]
        passages_with_golden = []
        for pid, psg in enumerate(passages):
            for aid, ans in enumerate(answers):
                try:
                    _ = regex.match(ans, psg)
                except:
                    continue
                else:
                    for m in regex.finditer(ans, psg):
                        ans_loc[pid] += [(m.start(), m.end())]
                        if pid not in passages_with_golden:
                            passages_with_golden += [pid]

        p['token_labels'] = p['input_ids'].new(p['input_ids'].shape).fill_(0)
        offset_mapping_with_golden = p['offset_mapping'][passages_with_golden]

        for pid, passage_offset in enumerate(offset_mapping_with_golden):
            for tid, token_offset in enumerate(passage_offset):
                for ans_offset in ans_loc[passages_with_golden[pid]]:
                    ans_begin, ans_end = ans_offset
                    token_begin, token_end = token_offset
                    if token_begin < ans_end and token_end > ans_begin:
                        p['token_labels'][passages_with_golden[pid]][tid] = 1

        token_labels.append(p['token_labels'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    token_labels = torch.cat(token_labels, dim=0)

    return passage_ids, passage_masks.bool(), token_labels


# 用于数据批处理（batch processing）的一个工具，通常用于准备输入数据，以便将其输入到模型中
class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20, add_loss=None, extra_decoder_inputs=False, n_context=1):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.add_loss = add_loss
        self.extra_decoder_inputs = extra_decoder_inputs
        self.n_context = n_context

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding='max_length',
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        if self.extra_decoder_inputs:
            extra_decoder_inputs = ['The answer to question ' + ex['question'] + ' is ' + ex['target'] for ex in batch]

        #  将问题和若干上下文（passages）合并成一个新的文本列表
        def append_question(example, n_context):
            if example['passages'] is None:
                return [example['question']]

            return_list = []
            candidate_passages = example['passages']
            source_buggy_code_segments = [example['question']]
            
            return_list = source_buggy_code_segments + candidate_passages

            while len(return_list) < (n_context+1):
                return_list = return_list + source_buggy_code_segments
            
            n_context = n_context+1
            return_list = return_list[0:n_context]
            return return_list


        text_passages = [append_question(example, self.n_context) for example in batch]

        if self.add_loss == None or self.add_loss in ["binary", "mse"]:
            try:
                passage_ids, passage_masks = encode_passages(text_passages,
                                                         self.tokenizer,
                                                         self.text_maxlength)
            except:
                print("index:", index)
                print("target:", target)
                print("text_passages:", text_passages)
                print()
                passage_ids, passage_masks = encode_passages(text_passages,
                                                         self.tokenizer,
                                                         self.text_maxlength)
            golden = torch.tensor([ex['golden'] for ex in batch])
        elif self.add_loss in ["span"]:
            passage_ids, passage_masks, answer_token_ids = encode_passages_spans(
                text_passages,
                [ex['target'] for ex in batch],
                self.tokenizer,
                self.text_maxlength)
            golden = answer_token_ids
        elif self.add_loss in ["group_tagger"]:
            passage_ids, passage_masks, answer_token_ids = encode_passages_group_tagger(
                text_passages,
                [ex['candidate_answers'] for ex in batch],
                self.tokenizer,
                self.text_maxlength,
            )
            golden = answer_token_ids
        elif self.add_loss in ["binary_token"]:
            passage_ids, passage_masks, answer_token_ids = encode_passages_group_tagger(
                text_passages,
                [ex['candidate_answers'] for ex in batch],
                self.tokenizer,
                self.text_maxlength,
            )
            golden_psg = torch.tensor([ex['golden'] for ex in batch])
            golden = torch.cat([golden_psg.unsqueeze(-1), answer_token_ids], dim=-1)
        else:
            raise ValueError("Loss {} is not used".format(self.add_loss))

        return (index, target_ids, target_mask, passage_ids, passage_masks, golden)


def append_question(example, n_context):
    """
    将问题和上下文拼接为一个长序列。
    """
    if example['passages'] is None:
        return example['question']

    # 将问题和上下文合并
    candidate_passages = example['passages']
    source_buggy_code_segments = [example['question']]
    combined_text = " ".join(source_buggy_code_segments + candidate_passages[:n_context])

    return combined_text


# 从指定路径加载数据，并对数据进行处理，最终返回一个包含数据样本的列表
def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()


    return examples


