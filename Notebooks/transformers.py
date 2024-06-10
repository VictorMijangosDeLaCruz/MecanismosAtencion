import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

def vocab():
    vocab = defaultdict()
    vocab.default_factory = lambda: len(vocab)
    return vocab

def index(corpus, voc, split=False):
    for sent in corpus:
        if split == True:
            sent = sent.split()
        else:
            pass
        yield torch.tensor([voc[w.lower()] for w in sent], dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, warmup=10000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = warmup**(2*torch.arange(0, d_model, 2)/d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, n):
        
        return self.pe[:n, :self.d_model]
        
class Encoding(nn.Module):
    def __init__(self, vocab_size, d_model, scale=True):
        super(Encoding, self).__init__()
        self.d_model = d_model
        self.scale = scale
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model=d_model)
        
    def forward(self, x):
        if self.scale:
            encoding = np.sqrt(self.d_model)*self.emb(x) + self.pe(x.size(0))
        else:
            encoding = self.emb(x) + self.pe(x.size(0))
        
        return encoding
        
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        # Capas de proyecciones
        self.d_model = d_model
        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K  = nn.Linear(d_model, d_model, bias=False)
        self.V  = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        # Proyección de los datos
        query,key,value = self.Q(x),self.K(x),self.V(x)
        # Cálculo de pesos de atención
        scores = torch.matmul(query, key.T)/np.sqrt(self.d_model)
        p_attn = torch.nn.functional.softmax(scores, dim = -1)
        #Suma ponderada
        Vs = torch.matmul(p_attn, value).reshape(x.shape)
        
        return Vs, p_attn

class MaskAttention(nn.Module):
    #Atención enmascarando subsecuentes
    def __init__(self, d_model):
        super(MaskAttention, self).__init__()
        self.d_model = d_model
        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        query, key, value = self.Q(x), self.K(x), self.V(x)
        scores = torch.matmul(query, key.T)/np.sqrt(self.d_model)
        #Enmascaramiento de los scores
        mask  = self.masking(x)
        scores = scores.masked_fill(mask == 0, -1e9)
        att = nn.functional.softmax(scores, dim=-1)
        h = torch.matmul(att, value)

        return h, att

    def masking(self, x):
        #Creación de la máscara
        n = x.size(0)
        subsequent_mask = np.triu(np.ones((n, n)), k=1).astype('uint8')
        
        return torch.from_numpy(subsequent_mask) == 0

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a*(x - mean)/(std + self.eps) + self.b
    
class Residual(nn.Module):
    def __init__(self, size):
        super(Residual, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, layer):
        return x + layer(self.norm(x))


class NoamOptimizer:
    def __init__(self, parameters, d_model, warmup=40000, init_lr=0, eps=1e-9, decay=0.01):
        #optimizador
        self.optimizer = torch.optim.Adam(parameters, lr=init_lr, betas=(0.9, 0.98), eps=eps, weight_decay=decay)
        self._step = 0
        self.warmup = warmup
        self.model_size = d_model
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self):
        step = self._step
        lr_step = self.model_size**(-0.5) * min(step**(-0.5), step*self.warmup**(-1.5))
        return lr_step

    def zero_grad(self):
        self.optimizer.zero_grad()
