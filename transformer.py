import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import spacy
import GPUtil
import warnings
import wandb
import logging
from datetime import datetime
from torchtext.data.metrics import bleu_score
import argparse

warnings.filterwarnings("ignore")

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        #print(f"Attention - scores shape: {scores.shape}, mask shape: {mask.shape}")
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)




def make_model(args, src_vocab, tgt_vocab):
    c = copy.deepcopy
    attn = MultiHeadedAttention(args.n_heads, args.d_model, args.dropout)
    ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
    position = PositionalEncoding(args.d_model, args.dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(args.d_model, c(attn), c(ff), args.dropout), args.n_layers),
        Decoder(DecoderLayer(args.d_model, c(attn), c(attn), c(ff), args.dropout), args.n_layers),
        nn.Sequential(Embeddings(args.d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(args.d_model, tgt_vocab), c(position)),
        Generator(args.d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model




class Batch:
    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


class TrainState:
    step: int = 0  
    accum_step: int = 0  
    samples: int = 0  
    tokens: int = 0  

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def run_epoch_wandb(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
    logger=None,
    epoch=0,
    gpu=0
):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        
        # 记录批次指标到wandb
        if mode == "train" and gpu == 0 and i % 10 == 0:  # 每10个批次记录一次
            wandb.log({
                "batch_loss": loss / batch.ntokens,
                "batch_step": train_state.step,
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            
            log_message = (
                f"Epoch {epoch} | GPU {gpu} | Step: {i:6d} | "
                f"Accum Step: {n_accum:3d} | Loss: {loss / batch.ntokens:6.2f} | "
                f"Tokens/Sec: {tokens / elapsed:7.1f} | LR: {lr:6.1e}"
            )
            
            if logger:
                logger.info(log_message)
            else:
                print(log_message)
                
            start = time.time()
            tokens = 0
            
        del loss
        del loss_node
        
    return total_loss / total_tokens, train_state



def calculate_bleu(model, dataloader, vocab_src, vocab_tgt, device, max_len=72):
    model.eval()
    translations = []
    references = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            src, tgt = batch[0].to(device), batch[1].to(device)
            src_mask = (src != vocab_src["<blank>"]).unsqueeze(-2)
            
            print(f"Batch {batch_idx}: Processing {src.size(0)} samples")
            
            # 使用贪心解码生成翻译
            output = greedy_decode(model, src, src_mask, max_len, vocab_tgt["<s>"], vocab_tgt)
            
            # 将输出转换为文本
            for i in range(output.size(0)):
                pred_tokens = []
                for idx in output[i]:
                    token_idx = idx.item()
                    if token_idx == vocab_tgt["</s>"]:
                        break
                    if token_idx not in [vocab_tgt["<s>"], vocab_tgt["<blank>"]]:
                        pred_tokens.append(vocab_tgt.get_itos()[token_idx])
                translations.append(pred_tokens)
                
                # 参考翻译（目标文本）
                ref_tokens = []
                for idx in tgt[i]:
                    token_idx = idx.item()
                    if token_idx == vocab_tgt["</s>"]:
                        break
                    if token_idx not in [vocab_tgt["<s>"], vocab_tgt["<blank>"]]:
                        ref_tokens.append(vocab_tgt.get_itos()[token_idx])
                references.append([ref_tokens])
    
    # 计算BLEU分数
    bleu = bleu_score(translations, references)
    return bleu, translations, references

def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


def greedy_decode(model, src, src_mask, max_len, start_symbol, vocab_tgt):
    #print(f"src shape: {src.shape}")  
    #print(f"src_mask shape: {src_mask.shape}")  

    batch_size = src.size(0)
    memory = model.encode(src, src_mask)
    
    # 初始化目标序列，为整个批次创建开始符号
    ys = torch.zeros(batch_size, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len - 1):
        # 创建目标序列掩码
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        #print(f"tgt_mask shape: {tgt_mask.shape}")
        
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        
        # 添加新单词到序列
        ys = torch.cat([
            ys, 
            next_word.unsqueeze(1)
        ], dim=1)
        
        # 检查是否所有序列都生成了结束符号
        if (next_word == vocab_tgt["</s>"]).all():
            break
            
    return ys

    
def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def build_vocabulary(spacy_de, spacy_en,data_dir):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)
    
    # 文件路径
    train_src_file = os.path.join(data_dir, "train.de")
    train_tgt_file = os.path.join(data_dir, "train.en")
    val_src_file = os.path.join(data_dir, "val.de")
    val_tgt_file = os.path.join(data_dir, "val.en")
    test_src_file = os.path.join(data_dir, "test.de")
    test_tgt_file = os.path.join(data_dir, "test.en")
    train_data = list(zip(open(train_src_file, encoding="utf-8"), open(train_tgt_file, encoding="utf-8")))
    val_data = list(zip(open(val_src_file, encoding="utf-8"), open(val_tgt_file, encoding="utf-8")))
    test_data = list(zip(open(test_src_file, encoding="utf-8"), open(test_tgt_file, encoding="utf-8")))
    data=train_data + val_data + test_data
    print("Building German Vocabulary ...")
    
    # 构建德语词汇表
    vocab_src = build_vocab_from_iterator(
        yield_tokens(data, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    
    # 构建英语词汇表
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(data, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en,data_dir):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en,data_dir)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt



def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    data_dir,
    batch_size=12000,
    max_padding=128,
):
    
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    
    # 本地文件路径
    train_src_file = os.path.join(data_dir, "train.de")
    train_tgt_file = os.path.join(data_dir, "train.en")
    val_src_file = os.path.join(data_dir, "val.de")
    val_tgt_file = os.path.join(data_dir, "val.en")
    test_src_file = os.path.join(data_dir, "test.de")
    test_tgt_file = os.path.join(data_dir, "test.en")
    def load_data_from_file(src_file, tgt_file):
        with open(src_file, 'r', encoding='utf-8') as src_f, open(tgt_file, 'r', encoding='utf-8') as tgt_f:
            return list(zip(src_f, tgt_f))  # 每个元素是 (src_sentence, tgt_sentence) 的元组

    #使用 yield_tokens 处理数据
    train_data = load_data_from_file(train_src_file, train_tgt_file)
    val_data = load_data_from_file(val_src_file, val_tgt_file)
    test_data = load_data_from_file(test_src_file, test_tgt_file)
    # 加载数据并创建数据迭代器
    print("Building German Vocabulary ...")

    
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader, test_dataloader

def train_worker(
    args,
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    data_dir,
):
    
    # 设置日志和wandb
    logger = setup_logging(args, gpu)
    logger.info(f"Train worker process using GPU: {gpu} for training")

    # 设置设备
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    pad_idx = vocab_tgt["<blank>"]
    model = make_model(args, len(vocab_src), len(vocab_tgt))
    model.to(device)
    module = model
    is_main_process = True

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.to(device)

    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        data_dir=args.data_dir,
        batch_size=args.batch_size // ngpus_per_node,
        max_padding=args.max_padding,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.base_lr, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, args.d_model, factor=1, warmup=args.warmup
        ),
    )
    train_state = TrainState()

    if is_main_process:
        wandb.watch(model, criterion, log="all", log_freq=100)

    for epoch in range(args.num_epochs):
        model.train()
        logger.info(f"[GPU{gpu}] Epoch {epoch} Training ====")
        train_loss, train_state = run_epoch_wandb(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=args.accum_iter,
            train_state=train_state,
            logger=logger,
            epoch=epoch,
            gpu=gpu
        )

        # 记录训练指标到wandb
        if is_main_process:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "train_tokens": train_state.tokens,
                "train_samples": train_state.samples
            })

        GPUtil.showUtilization()

        if is_main_process:
            filename = f"{args.file_prefix}{epoch:02d}.pt"
            file_path = os.path.join(args.save_dir, filename)
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(module.state_dict(), file_path)
            
            # 记录模型到wandb
            wandb.save(file_path)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 验证阶段
        logger.info(f"[GPU{gpu}] Epoch {epoch} Validation ====")
        model.eval()
        val_loss, _ = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            None,  
            None,
            mode="eval",
        )
        logger.info(f"Validation Loss: {val_loss}")
        
        # 记录验证指标到wandb
        if is_main_process:
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 训练完成后进行测试
    logger.info("训练完成，开始在测试集上评估模型...")
    test_loss, bleu_score = test_model(
        model, test_dataloader, vocab_src, vocab_tgt, device, logger
    )
    
    # 记录测试结果到wandb
    if is_main_process:
        wandb.log({
            "test_loss": test_loss,
            "bleu_score": bleu_score * 100  # 转换为百分比
        })

    if is_main_process:
        file_path = f"{args.file_prefix}final.pt"
        torch.save(module.state_dict(), file_path)
        wandb.save(file_path)
        wandb.finish()
        
    logger.info("训练和测试完成!")



def test_model(model, test_dataloader, vocab_src, vocab_tgt, device, logger=None):
    logger.info("开始测试模型...")
    
    # 计算测试损失
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=vocab_tgt["<blank>"], smoothing=0.1)
    criterion.to(device)
    
    test_loss, _ = run_epoch(
        (Batch(b[0], b[1], vocab_tgt["<blank>"]) for b in test_dataloader),
        model,
        SimpleLossCompute(model.generator, criterion),
        None,
        None,
        mode="eval"
    )
    
    # 计算BLEU分数
    bleu, translations, references = calculate_bleu(
        model, test_dataloader, vocab_src, vocab_tgt, device
    )
    
    # 记录结果
    logger.info(f"测试损失: {test_loss:.4f}")
    logger.info(f"BLEU分数: {bleu*100:.2f}")
    
    # 记录一些示例翻译
    logger.info("\n=== 翻译示例 ===")
    for i in range(min(3, len(translations))):
        logger.info(f"参考翻译: {' '.join(references[i][0])}")
        logger.info(f"模型生成: {' '.join(translations[i])}")
        logger.info("---")
    
    return test_loss, bleu


def run_test(args, model_path=None):
    logger = setup_logging(args, 1)
    logger.info("开始独立测试...")
    
    # 加载tokenizer和词汇表
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, args.data_dir)
    
    # 加载模型
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = make_model(args,len(vocab_src), len(vocab_tgt))
    
    if model_path is None:
        model_path = f"{args.file_prefix}final.pt"
    
    if exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"已加载模型: {model_path}")
    else:
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    model.to(device)
    
    # 创建测试数据加载器
    _, _, test_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_padding=args.max_padding
    )
    
    # 运行测试
    test_loss, bleu_score = test_model(
        model, test_dataloader, vocab_src, vocab_tgt, device, logger
    )
    
    logger.info(f"最终测试结果 - 损失: {test_loss:.4f}, BLEU: {bleu_score*100:.2f}")
    
    return test_loss, bleu_score



def setup_logging(args, gpu=1):
    # 创建日志目录
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志格式
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}_gpu{gpu}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(f"Transformer_Train_GPU{gpu}")
    
    # 初始化wandb（仅在主进程）
    if gpu == 1:  # 只在主GPU上初始化wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),  # 使用vars将args转换为字典
            dir=log_dir  # wandb日志也保存到同一目录
        )
        logger.info(f"W&B initialized with run ID: {wandb.run.id}")
    else:
        wandb.init(mode="disabled")  # 其他GPU禁用wandb
    
    return logger

def setup_args():
    parser = argparse.ArgumentParser(description='Transformer 训练和测试脚本')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='./multi30k', help='数据目录路径')
    parser.add_argument('--save_dir', type=str, default='./save_pt', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./training_logs', help='日志目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=8, help='训练轮数')
    parser.add_argument('--accum_iter', type=int, default=10, help='梯度累积步数')
    parser.add_argument('--base_lr', type=float, default=1.0, help='基础学习率')
    parser.add_argument('--max_padding', type=int, default=72, help='最大填充长度')
    parser.add_argument('--warmup', type=int, default=3000, help='预热步数')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_layers', type=int, default=6, help='编码器/解码器层数')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='丢弃率')
    
    # 实验跟踪
    parser.add_argument('--wandb_project', type=str, default='transformer-multi30k', help='W&B项目名')
    parser.add_argument('--run_name', type=str, default='transformer_baseline', help='实验名称')
    parser.add_argument('--file_prefix', type=str, default='multi30k_model_', help='模型文件前缀')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'], 
                       help='运行模式: train(训练), test(测试), both(先训练后测试)')
    parser.add_argument('--model_path', type=str, default=None, help='测试时模型路径')
    
    # 系统参数
    parser.add_argument('--distributed', action='store_true', help='是否使用分布式训练')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU ID')
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_args()
    
    # 创建必要的目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.mode in ["train", "both"]:
        spacy_de, spacy_en = load_tokenizers()
        vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, args.data_dir)
        model_path = f"{args.file_prefix}final.pt"
        
        if not exists(model_path):
            train_worker(args,args.gpu, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, args.data_dir)
        else:
            print(f"模型已存在: {model_path}，跳过训练")
    
    if args.mode in ["test", "both"]:
        run_test(args, args.model_path)
    
