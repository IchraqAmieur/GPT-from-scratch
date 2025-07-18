import torch
from torch.utils.data import DataLoader
from dataset import CharDataset
from model import GPT
import os

# ------------------- Hyperparameters -------------------
torch.manual_seed(1337)

block_size = 128
batch_size = 64
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.1
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 0  # For Windows compatibility

# ------------------- Data Preparation -------------------
dataset = CharDataset(block_size=block_size)
vocab_size = dataset.vocab_size
chars, stoi, itos = dataset.get_vocab()

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# ------------------- Model Setup -------------------
model = GPT(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    device=device
)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ------------------- Training Loop -------------------
model.train()
for iter_num in range(1, max_iters + 1):
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break  # Only one batch per iteration (for simplicity)

    if iter_num % eval_interval == 0 or iter_num == 1:
        print(f"Iter {iter_num}/{max_iters} | Loss: {loss.item():.4f}")

    # Save checkpoint at the end
    if iter_num == max_iters:
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stoi': stoi,
            'itos': itos,
            'vocab_size': vocab_size
        }, f'checkpoints/gpt_hermione.pt')
        print('Checkpoint saved!')

# ------------------- Text Generation -------------------
def generate_text(prompt, max_new_tokens=300, temperature=0.8, top_k=10):
    model.eval()
    idx = torch.tensor([dataset.encode(prompt)], dtype=torch.long).to(device)
    out_idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    out_text = dataset.decode(out_idx[0].tolist())
    return out_text

if __name__ == '__main__':
    prompt = input("Enter a prompt for Hermione: ")
    print("\n--- Hermione-style text ---\n")
    print(generate_text(prompt)) 