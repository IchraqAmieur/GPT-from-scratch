import torch
from dataset import CharDataset
from model import GPT

# Load the dataset to get vocabulary information
dataset = CharDataset(block_size=128)
vocab_size = dataset.vocab_size

# Load the saved checkpoint
checkpoint = torch.load('checkpoints/gpt_hermione.pt')

# Initialize the model with the same parameters
model = GPT(
    vocab_size=vocab_size,
    block_size=128,
    n_embd=256,
    n_head=4,
    n_layer=4,
    dropout=0.1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Load the saved state
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Function to generate text
def generate_text(prompt, max_new_tokens=300, temperature=0.8, top_k=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    idx = torch.tensor([dataset.encode(prompt)], dtype=torch.long).to(device)
    out_idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    out_text = dataset.decode(out_idx[0].tolist())
    return out_text

# Test the model
prompt = input("Enter a prompt: ")
print("\n--- Hp-style text ---\n")
print(generate_text(prompt))