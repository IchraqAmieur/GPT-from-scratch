import torch

INPUT_FILE = 'hp_full_script_clean.txt'
DATA_OUT = 'char_data.pt'
STOI_OUT = 'stoi.pt'
ITOS_OUT = 'itos.pt'

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

# Build vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode text
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

torch.save(data, DATA_OUT)
torch.save(stoi, STOI_OUT)
torch.save(itos, ITOS_OUT)

print(f"Vocab size: {vocab_size}")
print(f"Data tensor shape: {data.shape}")
print(f"Saved encoded data to {DATA_OUT}, stoi to {STOI_OUT}, itos to {ITOS_OUT}") 