# EEEGPT: A Nano Language Model for Electrical and Electronics Engineering
# This script builds a small-scale transformer model (~1M parameters) tailored for EEE knowledge.
# It includes dataset preparation, tokenization, model definition, training, evaluation, and inference.

import os
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import pdfplumber
import requests
from bs4 import BeautifulSoup

# -----------------------------------
# Step 1: Environment Setup
# -----------------------------------

# Create directory structure for the project
def setup_directories():
    project_root = 'eee_gpt'
    dirs = [
        os.path.join(project_root, 'data', 'raw'),
        os.path.join(project_root, 'data', 'processed'),
        os.path.join(project_root, 'src'),
        os.path.join(project_root, 'models'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directory structure created at:", project_root)

# -----------------------------------
# Step 2: Prepare Dataset (EEE Corpus)
# -----------------------------------

# Extract text from a PDF file (e.g., open-source textbook)
def extract_pdf_text(pdf_path, output_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ''
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Extracted text from {pdf_path} to {output_path}")
    except Exception as e:
        print(f"Error extracting PDF: {e}")

# Scrape text from a web page (e.g., MIT OpenCourseWare)
def scrape_web_text(url, output_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Scraped text from {url} to {output_path}")
    except Exception as e:
        print(f"Error scraping web page: {e}")

# Concatenate and clean text files
def preprocess_corpus(raw_dir, output_path):
    all_text = ''
    for filename in os.listdir(raw_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(raw_dir, filename), 'r', encoding='utf-8') as f:
                all_text += f.read() + '\n'
    # Basic cleaning: remove extra whitespace
    all_text = ' '.join(all_text.split())
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(all_text)
    print(f"Preprocessed corpus saved to {output_path}")

# Split corpus into training and validation sets
def split_corpus(corpus_path, train_path, val_path, split_ratio=0.9):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    split_idx = int(split_ratio * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_text)
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_text)
    print(f"Split corpus into {train_path} and {val_path}")

# Generate synthetic Q&A data if real data is limited
def generate_synthetic_data(output_path):
    synthetic_qa = """
    Q: What is Ohm's Law?
    A: Ohm's Law states that the current through a conductor between two points is directly proportional to the voltage across the two points, given by V = IR, where V is voltage, I is current, and R is resistance.

    Q: What is a transformer in electrical engineering?
    A: A transformer is a device that transfers electrical energy between two or more circuits through electromagnetic induction, typically used to step up or step down voltage levels.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(synthetic_qa)
    print(f"Generated synthetic data at {output_path}")

# -----------------------------------
# Step 3: Tokenization (Byte-Pair Encoding)
# -----------------------------------

# Train a BPE tokenizer
def train_tokenizer(train_file, tokenizer_path, vocab_size=5000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2)
    tokenizer.train(files=[train_file], trainer=trainer)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer trained and saved to {tokenizer_path}")
    return tokenizer

# Dataset class for tokenized sequences
class TextDataset(Dataset):
    def __init__(self, text_file, tokenizer, seq_len):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokens = tokenizer.encode(text).ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.seq_len]
        y = self.tokens[idx+1:idx+self.seq_len+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# -----------------------------------
# Step 4: Model Architecture (Nano GPT-like)
# -----------------------------------

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len):
        super().__init__()
        # Token and positional embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(max_seq_len, d_model)
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        # Output layer with tied weights
        self.output = nn.Linear(d_model, vocab_size)
        self.output.weight = self.embedding.weight
        self.max_seq_len = max_seq_len

    def forward(self, x):
        batch_size, seq_len = x.size()
        # Add positional encodings
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        x = self.embedding(x) + self.positional_encoding(positions)
        # Causal mask for autoregressive modeling
        mask = generate_square_subsequent_mask(seq_len).to(x.device)
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, src_mask=mask)
        # Output logits
        logits = self.output(x)
        return logits

# Generate causal mask for transformer
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# -----------------------------------
# Step 5: Training Loop
# -----------------------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

        # Save checkpoint
        checkpoint_path = f'eee_gpt/models/model_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')

# -----------------------------------
# Step 6: Evaluation & Testing
# -----------------------------------

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            val_loss += loss.item()
    perplexity = math.exp(val_loss / len(val_loader))
    print(f'Perplexity: {perplexity:.2f}')
    return perplexity

# Generate text with the model
def generate_text(model, tokenizer, prompt, device, max_len=50):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    for _ in range(max_len):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())

# -----------------------------------
# Step 7: Inference CLI
# -----------------------------------

def run_inference(model, tokenizer, device):
    print("\nEEEGPT CLI: Type 'exit' to quit.")
    while True:
        prompt = input("Enter prompt (e.g., 'In electrical engineering, a transformer is'): ")
        if prompt.lower() == 'exit':
            break
        generated = generate_text(model, tokenizer, prompt, device)
        print(f"Generated: {generated}\n")

# -----------------------------------
# Step 8: Model Export
# -----------------------------------

def export_model(model, device, output_path):
    model.eval()
    # Create a dummy input for tracing
    dummy_input = torch.randint(0, 5000, (1, 256)).to(device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    print(f"Exported model to {output_path}")

# -----------------------------------
# Main Execution
# -----------------------------------

def main():
    # Configuration
    vocab_size = 5000
    seq_len = 256
    batch_size = 32
    d_model = 128
    n_layers = 2
    n_heads = 4
    d_ff = 512
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Setup directories
    setup_directories()

    # Step 2: Prepare dataset
    raw_dir = 'eee_gpt/data/raw'
    processed_dir = 'eee_gpt/data/processed'
    
    # Example: Extract PDF (replace with your PDF path)
    pdf_path = os.path.join(raw_dir, 'textbook.pdf')
    if os.path.exists(pdf_path):
        extract_pdf_text(pdf_path, os.path.join(raw_dir, 'textbook.txt'))
    else:
        print("No PDF found; generating synthetic data.")
        generate_synthetic_data(os.path.join(raw_dir, 'synthetic.txt'))

    # Example: Scrape web content
    web_url = 'https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/'
    scrape_web_text(web_url, os.path.join(raw_dir, 'notes.txt'))

    # Preprocess and split corpus
    corpus_path = os.path.join(processed_dir, 'full_corpus.txt')
    train_path = os.path.join(processed_dir, 'train.txt')
    val_path = os.path.join(processed_dir, 'val.txt')
    preprocess_corpus(raw_dir, corpus_path)
    split_corpus(corpus_path, train_path, val_path)

    # Step 3: Train tokenizer
    tokenizer_path = os.path.join(processed_dir, 'tokenizer.json')
    tokenizer = train_tokenizer(train_path, tokenizer_path, vocab_size)

    # Step 4: Create datasets and data loaders
    train_dataset = TextDataset(train_path, tokenizer, seq_len)
    val_dataset = TextDataset(val_path, tokenizer, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Step 5: Initialize model
    model = NanoGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=seq_len
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 6: Train model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

    # Step 7: Evaluate model
    evaluate_model(model, val_loader, criterion, device)

    # Step 8: Run inference CLI
    run_inference(model, tokenizer, device)

    # Step 9: Export model
    export_path = 'eee_gpt/models/eeegpt.pt'
    export_model(model, device, export_path)

if __name__ == "__main__":
    main()