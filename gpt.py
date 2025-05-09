import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""
HYPER PARAMETERS
"""
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 10000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-3
EVAL_ITERS = 200

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
Bigram Language Model
"""
class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size:int):
        super().__init__()
        # create an embedding matrix 
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, idx, targets=None):
        # get data from embedding space
        logits = self.token_embedding_table(idx) # (B:batch, T:context, C:embedding_dim)
        
        if targets is None:
            loss = None
        else:
            # dims of the batched logits 
            B,T,C = logits.shape
            # we want to squash the batches st we can evaluate with cross entropy
            logits = logits.view(B*T,C)
            targets = targets.view(B*T) # note targets is shape (B:batch, T:context)
            # use cross entropy to calculate the loss
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    # this generate function is a core part of how autoregressive models function
    def generate(self, idx, max_new_tokens:int):
        # idx is (B,T) array of indices in our context window
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
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
model = BigramLanguageModel(vocab_size)
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
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))