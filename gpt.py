import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""
HYPER PARAMETERS
"""
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
EVAL_ITERS = 200
N_EMBED = 128
N_LAYERS = 4
N_HEADS = 4
DROPOUT = 0.2

torch.manual_seed(1337)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


"""
Read in the Dataset
"""
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


"""
Character Level Tokenization
"""
# get the vocab size
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mappings
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# init encoder and decoder functions
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


"""
Split Data into Training and Val Sets
"""
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, 10% val
train_data, val_data = data[:n], data[n:]


"""
Parallelize data into batches
"""
def get_batch(split:str):
    """
    X: if BLOCK_SIZE (context) is n, then n tensors (of length 1 to n-1) will be created via a simple sliding window
    Y: the next (target) character  
    """
    # select split
    data = train_data if split == 'train' else val_data
    # find b random start points from 0 to len(data)-BLOCK_SIZE 
    ix = torch.randint(len(data)-BLOCK_SIZE, (BATCH_SIZE,))
    # sliding window algo
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1: i+BLOCK_SIZE+1] for i in ix])

    x,y = x.to(DEVICE), y.to(DEVICE)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # set model to eval mode

    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set model back to training mode
    return out


"""
Single Head Attention
"""

class Head(nn.Module):
    def __init__(self, head_size:int):
        super().__init__()

        # init attention components
        self.k = nn.Linear(N_EMBED, head_size, bias=False)
        self.q = nn.Linear(N_EMBED, head_size, bias=False)
        self.v = nn.Linear(N_EMBED, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE))) # BLOCK_SIZE == T

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B,T,C  = x.shape
        q,k  = self.q(x), self.k(x) # (B,T,C)

        w = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)

        w = w.masked_fill(self.tril[:T,:T] == 0, float("-inf")) # (B,T,T)
        w = F.softmax(w, dim=-1) # (B,T,T)
        w = self.dropout(w)

        v = self.v(x) # (B,T,head_size)
        out = w @ v  #(B,T, head_size)

        return out


"""
Multi head attention
"""
class MultHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # linear transformation for the final output
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

"""
MLP
"""
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        # instantiate layers
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed), # the 4*n_embed replicates the dim of the MLP in the ATTN all u need paper
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed), # projection layer (similar to multihead)
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

"""
Attention Block
"""
class Block(nn.Module):
    def __init__(self, n_embed:int, n_head:int):
        super().__init__()

        head_size = n_embed//n_head
        self.ma = MultHead(n_head, head_size)
        self.ff_mlp = FeedForward(n_embed)

        # layer normalization
        self.layern1 = nn.LayerNorm(n_embed)
        self.layern2 = nn.LayerNorm(n_embed)


    def forward(self, x):
        # the x + () part is a residual connection
        # allows the compution to fork off and add its contribution to the output
        x = x + self.ma(self.layern1(x))
        x = x + self.ff_mlp(self.layern2(x))
        return x


"""
Bigram Language Model
"""
class LM(nn.Module):
    def __init__(self):
        super().__init__()
        # create an embedding matrix 
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)

        self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEADS) for _ in range(N_LAYERS)])

        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape # Note idx and targets are both of dim (B,T)
        # get data from embedding space
        token_embed = self.token_embedding_table(idx) # (B,T,C)
        position_embed = self.position_embedding_table(torch.arange(T,device=DEVICE)) # (T,C)
        x = token_embed + position_embed # (B,T,C)
        for b in self.blocks:
            x = b(x)
        # x = self.sa_head(x) # (B,T,C)
        # x = self.ff_mlp(x) # (B,T,C)

        # since embedding dim < vocab size we need to use a linear layer (lm head) to take the embedding -> logits
        logits = self.lm_head(x) # (B:batch, T:time, vocab_size)

        if targets is None:
            loss = None
        else:
            # dims of the batched logits 
            B,T,C = logits.shape
            # we want to squash the batches st we can evaluate with cross entropy
            logits = logits.view(B*T,C)
            targets = targets.view(B*T) # note targets is shape (B:batch, T:time)
            # use cross entropy to calculate the loss
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    # this generate function is a core part of how autoregressive models function
    def generate(self, idx, max_new_tokens:int):
        # idx is (B,T) array of indices in our context window
        for _ in range(max_new_tokens):
            # crops idx to the last blocksize tokens (to prevent out of range)
            idx_cond = idx[:,-BLOCK_SIZE:]
            # get predictions
            logits, loss = self(idx_cond)
            # go to the last character so that we can get the nxt char prediction for that char
            logits = logits[:,-1,:] # (B,T,C) -> (B,C) @ last element
            probs = F.softmax(logits, dim=-1) # (B,C)
            # calculate the next idx from the distribution above
            next_idx = torch.multinomial(probs, num_samples=1) # (B,1)
            # concatenate this to idx so that we can predict the nxt char of this one
            # effectively the context gets larger and larger (even though this bigram model does not need it)
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)

        return idx


    
"""
Training Loop
"""
# init model
model = LM()
m = model.to(DEVICE)
# training loop
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)


for step in range(MAX_ITERS):

    if step%EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"{step}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    # get the batched data
    xb,yb = get_batch('train')
    # evaluate the loss
    logits, loss = m(xb,yb)
    # clear gradients
    optimizer.zero_grad(set_to_none=True)
    # backprop
    loss.backward()
    # update grads
    optimizer.step()
   #print(loss.item())


if __name__ == "__main__":
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))