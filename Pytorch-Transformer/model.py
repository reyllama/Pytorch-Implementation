import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

########### Reference #############
n_vocab= len(vocab)
n_seq= 256
n_layer= 6
d_model= 512
n_head= 8
d_head= 64
###################################

def positional_encoding(n_seq, d_model):
    def cal_angle(pos, i):
        return pos / 10000**(2*i/d_model)
    def init_table(pos):
        return [cal_angle(pos, i) for i in range(d_model)]

    table = np.array([init_table(pos) for pos in range(n_seq)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])

    return table

class scaled_dotproduct_attention(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.scale = 1/np.sqrt(d_head)

    def forward(self, Q, K, V, mask):
        score = torch.mm(Q, K.transpose(1, 0)).mul_(self.scale)
        if mask is not None:
            score.masked_fill_(mask, -1e9)
        attention_prob = F.softmax(score, dim=1)
        context = torch.mm(attention_prob, V)

        return context, attention_prob

class multihead_attention(nn.Module):
    def __init__(self, d_model, n_head, d_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.W_Q = nn.Linear(d_model, n_head*d_head)
        self.W_K = nn.Linear(d_model, n_head*d_head)
        self.W_V = nn.Linear(d_model, n_head*d_head)

        self.scaled_dot_attention = scaled_dotproduct_attention(d_head)
        self.linear = nn.Linear(n_head*d_head, d_model)

    def forward(self, Q, K, V, mask):
        batch_size = Q.size(0)
        Q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        K_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        V_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        context, attention_prob = self.scaled_dot_attention(Q_s, K_s, V_s, mask)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.n_head*self.d_head)
        output = self.linear(context)

        return output, attention_prob

def get_attention_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_q.data.eq(0)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask

def get_decoder_mask(seq):
    decoder_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    decoder_mask = decoder_mask.triu(diagonal=1)
    return decoder_mask

class FFNN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model*4, kernel_size=1)
        self.conv2 = nn.Conv2d(d_model*4, d_model, kernel_size=1)

    def forward(self, x):
        x = F.gelu(self.conv1(x.transpose(1,2)))
        x = self.conv2(x.transpose(1,2))
        return x


class encoder_layer(nn.Module):
    def __init__(self, d_model, n_head, d_head):
        super().__init__()
        self.multihead_attention = multihead_attention(d_model, n_head, d_head)
        self.layernorm = nn.LayerNorm(d_model)
        self.linear = FFNN(d_model)

    def forward(self, x, mask):
        out, attn_prob = self.multihead_attention(x,x,x, mask) # self-attention
        x = self.layernorm(x+out)
        out = self.linear(x)
        x = self.layernorm(x+out)

    return x, attn_prob

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_head, d_head, n_seq, n_vocab):
        super().__init__()
        self.enc_emb = nn.Embedding(n_vocab, d_model)
        table = torch.FloatTensor(positional_encoding(n_seq+1, d_model))
        self.pos_emb = nn.Embedding.from_pretrained(table, freeze=True)
        self.layers = nn.ModuleList([encoder_layer(d_model, n_head, d_head) for _ in range(n_layers)])

    def forward(self, x):
        positions = np.arange(x.size(1)).expand(x.size(0), x.size(1)).contiguous() + 1
        pos_mask = x.eq(0)
        positions.masked_fill_(pos_mask, 0)

        x = self.enc_emb(x) + self.pos_emb(positions)

        mask = get_attention_mask(x, 0) # Needs Function Definition

        attention_prob = []

        for layer in self.layers:
            x, attn_prob = layer(x, mask)
            attention_prob.append(attn_prob)

        return x, attention_prob

class decoder_layer(nn.Module):
    def __init__(self, d_model, n_head, d_head):
        super().__init__()
        self.multihead_attention = multihead_attention(d_model, n_head, d_head)
        self.multihead_attention2 = multihead_attention(d_model, n_head, d_head)
        self.layernorm = nn.LayerNorm(d_model)
        self.linear = FFNN(d_model)

    def forward(self, dec_input, enc_output, self_attention_mask, dec_enc_attention_mask):
        x, self_attention_prob = self.multihead_attention(dec_input, dec_input, dec_input, self_attention_mask)
        dec_input = self.layernorm(dec_input+x)
        x, dec_enc_attention_prob = self.multihead_attention2(dec_input, enc_output, enc_output, dec_enc_attention_mask)
        enc_output = self.layernorm(enc_output+x)
        x = self.linear(enc_output)
        output = self.layernorm(x+enc_output)

        return output, self_attention_prob, dec_enc_attention_prob

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_head, d_head, n_seq, n_vocab):
        super().__init__()
        self.dec_emb = nn.Embedding(n_vocab, d_model)
        table = positional_encoding(n_seq, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(table, freeze=True)
        self.layers = nn.ModuleList([decoder_layer(d_model, n_head, d_head) for _ in range(n_layers)])

    def forward(self, dec_in, enc_out, enc_in):
        positions = np.arange(dec_in.size(1)).expand(dec_in.size(0), dec_in.size(1)).contiguous()+1
        pos_mask = dec_in.eq(0)
        positions.masked_fill_(pos_mask, 0)

        dec_out = self.dec_emb(dec_in) + self.pos_emb(positions)
        dec_attention_pad_mask = get_attention_mask(dec_in, dec_in, 0)
        dec_attention_decoder_mask = get_decoder_mask(dec_in)
        dec_self_attention_mask = torch.gt((dec_attention_pad_mask+dec_attention_decoder_mask), 0)
        dec_enc_attention_mask = get_attention_mask(dec_in, enc_in, 0)

        self_attention_probs, dec_enc_attention_probs = [], []

        for layer in self.layers:
            dec_out, self_attention_prob, dec_enc_attention_prob = layer(dec_out, enc_out, dec_self_attention_mask, dec_enc_attention_mask)
            self_attention_probs.append(self_attention_prob)
            dec_enc_attention_probs.append(dec_enc_attention_prob)

        return dec_out, self_attention_probs, dec_enc_attention_probs

class Transformer(nn.Module):
    def __init__(self, n_layers, d_model, n_head, d_head, n_seq, n_vocab):
        super().__init__()
        self.encoder = Encoder(n_layers, d_model, n_head, d_head, n_seq, n_vocab)
        self.decoder = Decoder(n_layers, d_model, n_head, d_head, n_seq, n_vocab)

    def forward(self, enc_in, dec_in):
        enc_out, enc_self_attention_prob = self.encoder(enc_in)
        dec_out, dec_self_attention_prob, dec_enc_attention_prob = self.decoder(dec_in, enc_out, enc_in)

    return dec_out, enc_self_attention_prob, dec_self_attention_prob, dec_enc_attention_prob
