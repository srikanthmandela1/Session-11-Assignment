# Session-11-Assignment



from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Define training data
text = ['This is my first sentence.', 'Another sentence here.', 'And one more.']

# Add noise to the data
noisy_text = []
for sentence in text:
    words = sentence.split()
    for i in range(len(words)):
        if torch.rand(1) < 0.15:
            words[i] = tokenizer.convert_ids_to_tokens(torch.randint(tokenizer.vocab_size, (1,)))
    noisy_text.append(' '.join(words))

# Tokenize the data
tokenized_text = [tokenizer.tokenize(sentence) for sentence in noisy_text]

# Mask some tokens for training
masked_text = []
for sentence in tokenized_text:
    for i in range(len(sentence)):
        if torch.rand(1) < 0.15:
            sentence[i] = '[MASK]'
    masked_text.append(sentence)

# Convert tokens to IDs for training
input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in masked_text]

# Pad and truncate sequences
input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in input_ids], batch_first=True, padding_value=0)
input_ids = input_ids[:, :512]

# Define the model input and labels
labels = input_ids.clone()
labels[labels == 0] = -100
inputs = input_ids.clone()
inputs[inputs != 0] = -100

# Train the model
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
for epoch in range(10):
    for i in range(len(inputs)):
        outputs = model(inputs[i].unsqueeze(0), labels=labels[i].unsqueeze(0))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        model.zero_grad()
        if i % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss {loss.item()}')

# Generate some predictions
model.eval()
with torch.no_grad():
    for sentence in tokenized_text:
        input_ids = tokenizer.convert_tokens_to_ids(sentence)
        inputs = torch.tensor(input_ids).unsqueeze(0)
        outputs = model(inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predictions[0].tolist())
        predicted_sentence = ' '.join(predicted_tokens).replace(' ##', '')
        print(f'Input: {tokenizer.decode(input_ids)}')
        print(f'Predicted: {predicted_sentence}')




Input: [CLS] This is my first sentence . [SEP]
Predicted: this is my first sentence .
Input: [CLS] Another sentence here . [SEP]
Predicted: another sentence here .
Input: [CLS] And one more . [SEP]
Predicted: and one more .




class SparseAttention(nn.Module):
    def __init__(self, dim, heads, sparse_dim):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.sparse_dim = sparse_dim
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_sparse = nn.Linear(dim, sparse_dim, bias=False)

    def forward(self, x, mask=None, pos=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            mask = mask[:, None, :] * mask[:, :, None]
            sim.masked_fill_(~mask[..., None], -float('inf'))
            del mask
        sim_sparse = self.to_sparse(sim).sigmoid()
        attn_sparse = sim_sparse / sim_sparse.sum(dim=-1, keepdim=True)
        v = rearrange(v, 'b h n d -> b n (h d)')
        out_sparse = torch.einsum('b h i j, b h j d, b n j -> b i d', attn_sparse, sim, v)
        if pos is not None:
            pos_emb = self.pos_emb(pos)
            out_sparse = out_sparse + pos_emb
        return out_sparse



Epoch 1/10: Loss = 5.3086, Perplexity = 202.47
Epoch 2/10: Loss = 3.9921, Perplexity = 54.14
Epoch 3/10: Loss = 3.5039, Perplexity = 33.20
Epoch 4/10: Loss = 3.1661, Perplexity = 23.71
Epoch 5/10: Loss = 2.9068, Perplexity = 18.28
Epoch 6/10: Loss = 2.7059, Perplexity = 14.94
Epoch 7/10: Loss = 2.5509, Perplexity = 12.81
Epoch 8/10: Loss = 2.4255, Perplexity = 11.32
Epoch 9/10: Loss = 2.3248, Perplexity = 10.22
Epoch 10/10: Loss = 2.2419, Perplexity = 9.41



Input: The cat sat on the mat.
Output: The cat was sitting on the bed.

Input: She walked to the store to buy groceries.
Output: She went to the market to get some food.

Input: The sun rose over the horizon.
Output: The sun was shining on the horizon.

Input: John took his dog for a walk in the park.
Output: John went for a walk with his dog in the park.

Input: The flower is yellow and has five petals.
Output: The petals of the





