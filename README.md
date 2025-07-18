# Harry Potter GPT (Character-level Transformer)

A PyTorch implementation of a character-level GPT-style language model trained on the Harry Potter movie scripts. Inspired by the "Attention is All You Need" paper and modern transformer architectures.

## Features
- **Transformer-based architecture**: Implements multi-head self-attention, positional embeddings, and stacked transformer blocks.
- **Character-level modeling**: Learns to generate text one character at a time.
- **Autoregressive text generation**: Generate new Harry Potter-style text from a prompt.
- **Custom dataset**: Trained on cleaned Harry Potter scripts.
- **Easy training and inference scripts**: Train from scratch or generate text from a saved checkpoint.

## Project Structure
```
version2/
  |- model.py                # Transformer (GPT) model implementation
  |- train.py                # Training script
  |- test.py                 # Inference/generation script
  |- dataset.py              # PyTorch Dataset for character data
  |- prepare_char_dataset.py # Script to preprocess and encode data
  |- hp_full_script_clean.txt# Cleaned Harry Potter script (training data)
  |- char_data.pt, stoi.pt, itos.pt # Preprocessed data and vocab
```

## Setup
1. **Clone the repository**
2. **Install dependencies**
   - Python 3.7+
   - PyTorch (https://pytorch.org/)
   - (Optional) Jupyter for notebooks

   Install with pip:
   ```bash
   pip install torch
   ```

3. **Prepare the dataset**
   - Ensure `hp_full_script_clean.txt` is present in `version2/`.
   - Run the data preparation script:
     ```bash
     python prepare_char_dataset.py
     ```

## Training
Train the model from scratch:
```bash
python train.py
```
- Checkpoints will be saved in the `checkpoints/` directory.

## Generating Text
After training (or using a provided checkpoint):
```bash
python test.py
```
- Edit `test.py` to change the prompt or generation parameters.

## Customization
- **Hyperparameters**: Adjust model size, number of layers/heads, block size, etc. in `train.py` and `model.py`.
- **Data**: Replace `hp_full_script_clean.txt` with your own text for a different dataset.

## Credits
- Inspired by [Attention is All You Need](https://arxiv.org/abs/1706.03762) and [GPT](https://openai.com/research/publications/language-unsupervised).
- Implementation by [Your Name].

## License
This project is for educational purposes only. Not affiliated with J.K. Rowling or Warner Bros. 