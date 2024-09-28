import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# 嵌入表示层
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        # self.embedding = nn.Embedding(10, d_model, padding_idx=0)
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos][i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos][i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
                
        pe = pe.unsqueeze(0)
        # print(pe)
        # print(pe.shape) # torch.Size([1, 80, 512])
        # input()
        self.register_buffer('pe',pe)

    def forward(self, x):
        x = self.embedding(x)    # x.shape : batch_size, seq_len, d_model
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)
        # 这里的 requires_grad=False 表示在反向传播中不计算该变量的梯度，因为位置编码是固定的，不需要更新。
        x = (x + torch.tensor(self.pe[:,:seq_len], requires_grad=False)).cuda()
        return x


# 注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // heads  # bert 768 // 12 = 64
        self.h = heads  # 头数

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 上下文单词所对应的权重得分，形状是 seq_len, d_model × d_model, seq_len = seq_len, seq_len
        # 掩盖掉那些为了填补长度增加的单元，使其通过 softmax 计算后为 0
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # 进行线性操作划分为 h 个头， batch_size, seq_len, d_model -> batch_size, seq_len, h, d_k
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)  
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 矩阵转置  batch_size, seq_len, h, d_k -> batch_size, h, seq_len, d_k
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # 计算 attention
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # 连接多个头并输入到最后的线性层 (bs, h, seq_len, d_k) 转换为 (bs, seq_len, h, d_k)
        # .contiguous() 用于确保内存的连续性，方便后续的操作。
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


# 前馈层
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__(FeedForward, self)
        d_ff = d_model * 4
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)


# 残差连接与层归一化
class NormLayer(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(NormLayer, self).__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm




# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    # 归一化 -> attention -> 归一化 -> feedforward -> 归一化 
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    

def get_clones(module, N):
    """ 
    生成 N 个相同的模块。
    
    Parameters:
    - module: nn.Module，想要克隆的模块
    - N: 克隆的数量

    ModuleList 是 PyTorch 中的一个容器，可以方便地管理多个子模块。
    """
    return nn.ModuleList([module for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super(Encoder, self).__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = NormLayer(d_model)
        
    def forward(self, src, mask):
        x = self.embed(src)  # batch_size, seq_len -> batch_size, seq_len, d_model
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)



# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.norm_3 = NormLayer(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))  # q是x2，k是encoder的输出，v也是encoder的输出
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super(Decoder, self).__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = NormLayer(d_model)
    
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)   # 用的是trg
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


# 基于Transformer的编码器和解码器整体实现
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)  # 映射到 target_vocab的大小
    
    def forward(self,src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


def train_model():
    d_model = 512
    heads = 8
    N = 6
    # 英语翻译成法语
    EN_Vocab = {"hello": 1, "world": 2, "how": 3, "are": 4, "you": 5}
    FR_Vocab = {"bonjour": 1, "le": 2, "comment": 3, "va": 4, "tu": 5}
    src_vocab = len(EN_Vocab)
    trg_vocab = len(FR_Vocab)
    model = Transformer(src_vocab, trg_vocab, d_model, N, heads, 0.1)
    epochs = 10
    print_every = 100
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001,betas=(0.9, 0.98), eps=1e-9)
    
    start = time.time()
    temp = start
    total_loss = 0
    # 模型训练
    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            model = model.train()
            src = batch.English.transpose(0,1)
            trg = batch.French.transpose(0,1)
            # the French sentence we input has all words except
            # the last, as it is using each word to predict the next
            """
            trg_input = trg[:, :-1]:
            这里，我们创建了一个 trg_input，它包含目标序列（法语句子），但去掉了最后一个单词。这样做的目的是为了在训练过程中，使用当前单词来预测下一个单词。trg[:, :-1] 选择了从开始到倒数第二个位置的所有单词。
            targets = trg[:, 1:].contiguous().view(-1):
            trg[:, 1:] 从目标序列中去掉第一个单词，选择从第二个单词到最后一个单词，这些是我们希望模型预测的目标。
            """
            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            # create function to make masks using mask code above
            src_mask, trg_mask = create_masks(src, trg_input)
            preds = model(src, trg_input, src_mask, trg_mask)
            optim.zero_grad()
            loss = F.cross_entropy(preds.view(-1,preds.size(-1)), targets, ignore_index=target_pad)
            loss.backward()
            optim.step()
            total_loss += loss.data[0]
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d,iter = %d,loss = %.3f,%ds per %diters"% ((time.time() - start) // 60, epoch + 1,i + 1,loss_avg, time.time() - temp, print_every))
                total_loss = 0
                temp = time.time()


