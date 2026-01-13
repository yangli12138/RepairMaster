import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
from .modeling_t5 import T5ForConditionalGeneration
import faiss
import psutil
import os
import gc

class UnlimiformerT5(T5ForConditionalGeneration):
    def __init__(self, config, opt=None):
        super().__init__(config, opt)
        self.extra_decoder_inputs = opt.extra_decoder_inputs if opt else None
        self.wrap_encoder(opt=opt)
        self.split_psg_subset = opt.split_psg_subset if opt else False
        self.n_context = opt.n_context if opt else False
        self.output_attentions = opt.output_attentions if opt else False
        self.beam_size = opt.beam_size if opt else 4

        # 分块和窗口相关参数
        self.chunk_overlap = 0.5  # 相邻块重叠率
        self.model_encoder_max_len = 1024  # 最大编码长度
        self.window_margin = int(self.model_encoder_max_len * self.chunk_overlap / 2)

        # KNN 搜索相关初始化
        self.knn_index = None  # 初始化 FAISS 索引
        # 用于清理 FAISS 索引的步数间隔
        self.clean_interval = 50  # 每 50 步清理一次索引

    def initialize_knn_index(self, embedding_dim, n_neighbors=3):
        """
        初始化FAISS索引和KNN参数。
        """
        self.knn_index = faiss.IndexFlatL2(embedding_dim)  # 使用 L2 距离度量
        self.n_neighbors = n_neighbors  # 搜索时的K个邻居

    def get_memory_usage(self):
        """获取当前内存使用情况并打印"""
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # 返回MB
        # print(f"Current memory usage: {memory_usage:.2f} MB")  # 打印内存使用情况
        return memory_usage

    def update_knn_index(self, embeddings):
        """
        更新FAISS索引，并在内存占用过大时清理索引。
        根据步数每隔一定次数清理一次索引。
        """
        embeddings = embeddings.detach().cpu().numpy().astype(np.float32)

        # 如果是第一次初始化FAISS索引，则初始化它
        if self.knn_index is None:
            embedding_dim = embeddings.shape[1]
            self.initialize_knn_index(embedding_dim)

        # 将新的嵌入添加到FAISS索引中
        self.knn_index.add(embeddings)

    def knn_attention(self, query, embeddings, k=5):
        """
        使用 KNN 搜索来加权查询与上下文之间的注意力。
        """
        query = query.cpu().numpy().astype(np.float32)  # 转为 numpy 格式
        D, I = self.knn_index.search(query, k)  # 搜索最近的 K 个邻居
        # 返回邻居的索引和距离（D 是距离，I 是索引）
        return I, D

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(UnlimiformerT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, step=None, **kwargs):
        if input_ids is not None:
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)

            # 分块处理输入
            input_ids, attention_mask = self.chunked_input(input_ids, attention_mask)

        # 调用标准的前向传播
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # 获取当前批次的 logits
        logits = output.logits  # 使用 logits 而不是 last_hidden_state

        # 获取嵌入：可以选择取平均或选择最后的 logits
        embeddings = logits.mean(dim=1)  # 对每个样本的所有位置取均值 (batch_size, hidden_dim)

        # 或者选择最后的标记作为嵌入 (取最后一个位置的 logits)
        # embeddings = logits[:, -1, :]  # 使用最后一个标记作为嵌入

        # 更新 KNN 索引
        self.update_knn_index(embeddings)

        return output

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, add_loss=None, max_length=128):
        self.encoder.n_passages = input_ids.size(1)

        # 分块处理输入
        input_ids, attention_mask = self.chunked_input(input_ids, attention_mask)

        # 调用生成逻辑
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=self.beam_size,
            return_dict_in_generate=self.output_attentions
        )

    def chunked_input(self, input_ids, attention_mask):
        """
        将输入序列按上下文合并，并按窗口分块。
        """
        # 获取输入的维度
        batch_size, n_passages, passage_length = input_ids.shape

        # 展平为二维 (batch_size, total_length)
        input_ids = input_ids.view(batch_size, -1)
        attention_mask = attention_mask.view(batch_size, -1)

        total_length = input_ids.size(1)

        # 计算期望的总长度
        expected_length = n_passages * passage_length

        if total_length < expected_length:
            # 如果总长度不足，补零
            pad_length = expected_length - total_length
            input_ids = F.pad(input_ids, (0, pad_length))  # 在末尾补零
            attention_mask = F.pad(attention_mask, (0, pad_length))  # 对应的 mask 也补零
        elif total_length > expected_length:
            # 如果总长度多于期望值，截断多余部分
            input_ids = input_ids[:, :expected_length]
            attention_mask = attention_mask[:, :expected_length]

        return input_ids, attention_mask

    def wrap_encoder(self, use_checkpoint=False, opt=None):
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint, opt=opt)

    def unwrap_encoder(self):
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, False)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def overwrite_attention(self):
        """
        重写跨注意力机制，限制窗口内注意力。
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(self.custom_attention_forward, attn)

    @staticmethod
    def custom_attention_forward(self, input, mask=None, kv=None, position_bias=None, **kwargs):
        """
        自定义跨注意力，限制在滑动窗口范围内，并结合 KNN 信息。
        """
        bsz, qlen, dim = input.size()
        n_heads, d_heads = self.n_heads, self.d_kv
        klen = kv.size(1)

        q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)

        # 获取 KNN 权重
        I, D = self.knn_attention(q, k, k=self.n_neighbors)

        # 在 KNN 基础上加权
        knn_attention = torch.zeros_like(q)
        for i in range(bsz):
            for j in range(qlen):
                # 从 I 获取邻居的索引，D 获取距离
                for neighbor_idx, distance in zip(I[i], D[i]):
                    knn_attention[i, j] += k[neighbor_idx] * distance

        # 限制注意力范围
        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)
        if mask is not None:
            scores += mask

        if position_bias is not None:
            scores += position_bias

        scores += knn_attention  # 将 KNN 注意力加入到原始的注意力分数中

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
        output = self.o(output)
        return output

    def get_crossattention_scores(self, context_mask):
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder, use_checkpoint=False, opt=None):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)
        self.base_model_prefix = ""
        self.main_input_name = "input_ids"
        self.split_psg_subset = opt.split_psg_subset if opt is not None else False
        self.n_context = opt.n_context if opt is not None else False

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        重写 forward 方法以处理动态分块后的输入。
        """
        bsz, total_length = attention_mask.shape
        passage_length = total_length // self.n_passages  # 动态计算 passage_length

        if total_length % self.n_passages != 0:
            raise ValueError(
                f"Total length ({total_length}) is not divisible by number of passages ({self.n_passages})."
            )

        # 展平为 (bsz * n_passages, passage_length)
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)

        # 调用编码器
        outputs = self.encoder(input_ids, attention_mask, **kwargs)

        # 恢复为 (bsz, total_length, hidden_dim)
        outputs["last_hidden_state"] = outputs[0].view(bsz, total_length, -1)
        return outputs


class CheckpointWrapper(torch.nn.Module):
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block

def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output


