# Building EEEGPT: A Nano Language Model for Electrical Engineering

This guide provides a step-by-step process to build EEEGPT, a custom nanoscale language model tailored for Electrical and Electronics Engineering (EEE) knowledge, with approximately 100K–1M parameters, suitable for running on a local machine or Google Colab. Each step includes detailed code, explanations, and best practices, using PyTorch and a Byte-Pair Encoding (BPE) tokenizer.

## ✅ Step 1: Environment Setup

**Goal**: Set up the Python environment and project structure for building and training EEEGPT.

**What You’ll Do**:
- Install required Python libraries: `torch`, `tokenizers`, `numpy`.
- Create a directory structure for data, source code, and models.
- Verify the setup for compatibility with CPU or GPU (e.g., Colab).

**Code**:

1. **Create a Virtual Environment** (recommended for local setups):
```bash
python -m venv eee_gpt_env
source eee_gpt_env/bin/activate  # On Windows: eee_gpt_env\Scripts\activate
```

2. **Install Libraries**:
```bash
pip install torch tokenizers numpy
```
*Note*: For GPU support, install the appropriate PyTorch version with CUDA from the [PyTorch website](https://pytorch.org/get-started/locally/). In Colab, run `!pip install tokenizers numpy` in a notebook cell, as `torch` is typically pre-installed.

3. **Set Up Directory Structure**:
```python
import os

project_root = 'eee_gpt'
dirs = [
    os.path.join(project_root, 'data', 'raw'),
    os.path.join(project_root, 'data', 'processed'),
    os.path.join(project_root, 'src'),
    os.path.join(project_root, 'models'),
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

print("Directory structure created.")
```

**Explanation**:
- **Virtual Environment**: Isolates project dependencies, preventing conflicts with other Python projects.
- **Libraries**:
  - `torch`: Provides tools for building and training the transformer model.
  - `tokenizers`: Enables training a BPE tokenizer for EEE-specific text.
  - `numpy`: Supports numerical operations, such as data preprocessing.
- **Directory Structure**:
  - `data/raw/`: Stores unprocessed data (e.g., PDFs, HTML files).
  - `data/processed/`: Holds cleaned and split datasets (e.g., `train.txt`, `val.txt`).
  - `src/`: Contains source code (e.g., `model.py`, `train.py`).
  - `models/`: Saves trained model checkpoints.
- **Colab Note**: In Colab, skip the virtual environment and run `!pip install tokenizers numpy` in a cell. The directory structure can be created in the Colab file system.

**Expected Output**:
- Libraries installed successfully (verify with `pip list`).
- Directory structure created under `eee_gpt/` with subfolders `data/raw`, `data/processed`, `src`, and `models`.

**Best Practices**:
- Use a virtual environment for local development to manage dependencies.
- Check PyTorch compatibility with your hardware (CPU/GPU) to optimize training.
- Organize files early to streamline data processing and model training.

## ✅ Step 2: Prepare Dataset (EEE Corpus)

**Goal**: Collect and preprocess a text corpus of EEE knowledge for training EEEGPT.

**What You’ll Do**:
- Gather text from open-source EEE textbooks, research papers, and course notes.
- Clean and preprocess the text to remove noise (e.g., headers, tables).
- Split the corpus into training and validation sets.

**Code**:

1. **Install Additional Libraries** (for data extraction):
```bash
pip install pdfplumber requests beautifulsoup4
```

2. **Extract Text from PDFs** (e.g., open-source textbooks):
```python
import pdfplumber

with pdfplumber.open('data/raw/textbook.pdf') as pdf:
    text = ''
    for page in pdf.pages:
        text += page.extract_text() or ''

with open('data/raw/textbook.txt', 'w', encoding='utf-8') as f:
    f.write(text)
```

3. **Scrape Text from Web Pages** (e.g., course notes):
```python
import requests
from bs4 import BeautifulSoup

url = 'https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text(separator=' ', strip=True)

with open('data/raw/notes.txt', 'w', encoding='utf-8') as f:
    f.write(text)
```

4. **Concatenate and Clean Text**:
```python
import os

raw_dir = 'data/raw'
all_text = ''
for filename in os.listdir(raw_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(raw_dir, filename), 'r', encoding='utf-8') as f:
            all_text += f.read() + '\n'

# Basic cleaning (e.g., remove extra whitespace)
all_text = ' '.join(all_text.split())

with open('data/processed/full_corpus.txt', 'w', encoding='utf-8') as f:
    f.write(all_text)
```

5. **Split into Training and Validation Sets**:
```python
with open('data/processed/full_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

split_idx = int(0.9 * len(text))
train_text = text[:split_idx]
val_text = text[split_idx:]

with open('data/processed/train.txt', 'w', encoding='utf-8') as f:
    f.write(train_text)
with open('data/processed/val.txt', 'w', encoding='utf-8') as f:
    f.write(val_text)
```

**Explanation**:
- **Data Sources**: Collect text from open-source resources like [Open Textbook Library](https://open.umn.edu/opentextbooks/subjects/electrical), [All About Circuits](https://www.allaboutcircuits.com/textbook/), [MIT OpenCourseWare](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/), and [arXiv EESS](https://arxiv.org/list/eess/recent). Alternatively, use the [Electrical-engineering dataset](https://huggingface.co/datasets/STEM-AI-mtl/Electrical-engineering) from Hugging Face for Q&A prompts.
- **Text Extraction**: Use `pdfplumber` for PDFs and `BeautifulSoup` for HTML to extract clean text. Ensure you have permission to use the data.
- **Preprocessing**: Concatenate all text files and perform basic cleaning (e.g., removing extra whitespace). Advanced cleaning (e.g., removing headers, equations) may require custom regex or NLP tools.
- **Splitting**: Divide the corpus into 90% training and 10% validation to ensure sufficient data for training while reserving some for evaluation.

**Expected Output**:
- Text files in `data/raw/` from various sources.
- `full_corpus.txt` in `data/processed/` containing the concatenated text.
- `train.txt` and `val.txt` in `data/processed/` for training and validation.

**Data Sources**:
| Source | Type | URL |
|--------|------|-----|
| Open Textbook Library | Textbooks | [Electrical Engineering Textbooks](https://open.umn.edu/opentextbooks/subjects/electrical) |
| All About Circuits | Textbooks | [Textbook](https://www.allaboutcircuits.com/textbook/) |
| MIT OpenCourseWare | Course Notes | [EECS Courses](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/) |
| arXiv | Research Papers | [EESS Papers](https://arxiv.org/list/eess/recent) |
| Hugging Face | Q&A Dataset | [Electrical-engineering](https://huggingface.co/datasets/STEM-AI-mtl/Electrical-engineering) |

**Best Practices**:
- Verify the legal use of data sources to comply with licensing (e.g., Creative Commons).
- Use robust text cleaning to remove non-text elements like equations or figures.
- Ensure the corpus is diverse, covering topics like circuits, electromagnetics, and power systems.

## ✅ Step 3: Tokenization (Byte-Pair Encoding)

**Goal**: Train a BPE tokenizer on the EEE corpus and prepare tokenized data for model input.

**What You’ll Do**:
- Train a BPE tokenizer using the `tokenizers` library.
- Save the tokenizer for encoding and decoding text.
- Create a dataset class to provide tokenized sequences.

**Code**:

1. **Train the Tokenizer**:
```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Train on training data
trainer = trainers.BpeTrainer(vocab_size=5000, min_frequency=2)
tokenizer.train(files=["data/processed/train.txt"], trainer=trainer)

# Save tokenizer
tokenizer.save("data/processed/tokenizer.json")
```

2. **Create Dataset Class**:
```python
from torch.utils.data import Dataset
import torch

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
```

**Explanation**:
- **Tokenizer Training**: The `tokenizers` library trains a BPE tokenizer on `train.txt`, creating a vocabulary of 5000 tokens. The `min_frequency=2` ensures rare tokens are excluded.
- **Dataset Class**: The `TextDataset` class encodes the text file into token IDs and provides sequences of length `seq_len` for training. The target sequence is shifted by one token for next-token prediction.
- **BPE**: BPE merges frequent character pairs to create subword tokens, balancing vocabulary size and text coverage, ideal for domain-specific text like EEE.

**Expected Output**:
- `tokenizer.json` in `data/processed/` containing the trained tokenizer.
- `TextDataset` class ready to provide tokenized sequences for training.

**Best Practices**:
- Choose a vocabulary size (e.g., 5000) suitable for a small model to reduce memory usage.
- Test the tokenizer on sample EEE text to ensure it captures domain-specific terms.
- Save the tokenizer for consistent encoding during training and inference.

## ✅ Step 4: Model Architecture (Nano GPT-like)

**Goal**: Design a small transformer-based language model with ~100K–1M parameters.

**What You’ll Do**:
- Define a `NanoGPT` model with a decoder-only transformer architecture.
- Configure hyperparameters to keep the model lightweight.
- Verify the parameter count.

**Code**:
```python
import torch
import torch.nn as nn

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(max_seq_len, d_model)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)
        self.output.weight = self.embedding.weight  # Tie weights

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        x = self.embedding(x) + self.positional_encoding(positions)
        for layer in self.transformer_layers:
            x = layer(x, src_mask=generate_square_subsequent_mask(seq_len).to(x.device))
        logits = self.output(x)
        return logits

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
```

**Explanation**:
- **Hyperparameters**:
  - `vocab_size=5000`: Matches the tokenizer’s vocabulary.
  - `d_model=128`: Embedding dimension for tokens and positions.
  - `n_layers=2`: Number of transformer layers.
  - `n_heads=4`: Number of attention heads.
  - `d_ff=512`: Feed-forward network dimension (4 * d_model).
  - `max_seq_len=256`: Maximum sequence length for positional encoding.
- **Architecture**: A decoder-only transformer with token and positional embeddings, followed by transformer layers with causal self-attention, and a linear output layer sharing weights with the embedding layer.
- **Parameter Count**:
  - Embedding: 5000 * 128 = 640,000
  - Positional encoding: 256 * 128 = 32,768
  - Transformer layer: (4 * 128² + 2 * 128 * 512) ≈ 196,608
  - Total for 2 layers: 640,000 + 32,768 + 2 * 196,608 ≈ 1,065,984 parameters
- **Causal Mask**: Ensures the model only attends to previous tokens, suitable for language modeling.

**Expected Output**:
- A `NanoGPT` model instance with ~1M parameters, verifiable by:
```python
model = NanoGPT(vocab_size=5000, d_model=128, n_layers=2, n_heads=4, d_ff=512, max_seq_len=256)
print(sum(p.numel() for p in model.parameters()))  # ~1,065,984
```

**Architecture Diagram** (Text-based):
```
Input Tokens -> Embedding -> + Positional Encoding -> Transformer Layers (x2) -> Linear Output -> Logits
                     |                             |
                     v                             v
                [vocab_size, d_model]         [max_seq_len, d_model]
```

**Best Practices**:
- Tie embedding and output weights to reduce parameters.
- Use dropout (0.1) to prevent overfitting.
- Ensure `n_heads` divides `d_model` evenly (128 / 4 = 32).

## ✅ Step 5: Training Loop

**Goal**: Train the EEEGPT model on the tokenized dataset using a GPU or CPU.

**What You’ll Do**:
- Set up data loaders for training and validation.
- Define the loss function and optimizer.
- Implement the training loop with checkpointing.

**Code**:
```python
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# Data loaders
seq_len = 256
batch_size = 32
train_dataset = TextDataset('data/processed/train.txt', tokenizer, seq_len)
val_dataset = TextDataset('data/processed/val.txt', tokenizer, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NanoGPT(vocab_size=5000, d_model=128, n_layers=2, n_heads=4, d_ff=512, max_seq_len=seq_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Training loop
num_epochs = 10
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
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss / len(val_loader)}')

    # Save checkpoint
    torch.save(model.state_dict(), f'models/model_epoch{epoch+1}.pth')
```

**Explanation**:
- **Data Loaders**: Load batches of tokenized sequences with `batch_size=32` and `seq_len=256`, suitable for small models.
- **Loss Function**: Cross-entropy loss for next-token prediction.
- **Optimizer**: Adam with a learning rate of 3e-4, standard for transformer training.
- **Training Loop**: Train for 10 epochs, compute training and validation losses, and save model checkpoints.

**Expected Output**:
- Console output showing training and validation losses per epoch.
- Model checkpoints saved as `model_epochX.pth` in `models/`.

**Best Practices**:
- Use a small batch size for CPU compatibility; increase if using a GPU.
- Monitor validation loss to detect overfitting.
- Save checkpoints to resume training or select the best model.

## ✅ Step 6: Evaluation & Testing

**Goal**: Evaluate the model’s performance and test its text generation capabilities.

**What You’ll Do**:
- Calculate perplexity on the validation set.
- Implement text generation with a prompt.

**Code**:
```python
import math

# Calculate perplexity
model.eval()
val_loss = 0
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        val_loss += loss.item()
perplexity = math.exp(val_loss / len(val_loader))
print(f'Perplexity: {perplexity}')

# Text generation
def generate_text(model, tokenizer, prompt, max_len=50):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    for _ in range(max_len):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())

prompt = "In electrical engineering, a transformer is"
generated = generate_text(model, tokenizer, prompt)
print(f'Generated: {generated}')
```

**Explanation**:
- **Perplexity**: Measures how well the model predicts the validation set; lower is better. Computed as `exp(average_loss)`.
- **Text Generation**: Uses greedy decoding to generate text by selecting the most likely next token iteratively.
- **Prompt Testing**: Test with EEE-specific prompts to verify domain knowledge.

**Expected Output**:
- Perplexity value (e.g., 50–100 for a small model, depending on data quality).
- Generated text continuing the prompt, e.g., describing transformers in EEE context.

**Best Practices**:
- Use perplexity to compare model performance across training runs.
- Test multiple prompts to assess the model’s understanding of EEE concepts.
- Consider sampling methods (e.g., top-k) for more diverse generation.

## ✅ Step 7: Fine-tuning & Optimization (Optional)

**Goal**: Fine-tune the model on specific EEE subtopics (e.g., control systems, power electronics).

**What You’ll Do**:
- Collect a smaller dataset for a specific subtopic.
- Fine-tune the model with a lower learning rate.
- Optimize for performance (e.g., quantization).

**Code**:
```python
# Example: Fine-tune on control systems Q&A dataset
fine_tune_dataset = TextDataset('data/processed/control_systems.txt', tokenizer, seq_len)
fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Lower learning rate
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in fine_tune_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Fine-tune Epoch {epoch+1}, Loss: {total_loss / len(fine_tune_loader)}')
```

**Explanation**:
- **Dataset**: Use a targeted dataset, e.g., Q&A from the [Electrical-engineering dataset](https://huggingface.co/datasets/STEM-AI-mtl/Electrical-engineering).
- **Fine-tuning**: Train with a smaller learning rate to adapt the model without catastrophic forgetting.
- **Optimization**: Consider techniques like quantization (e.g., `torch.quantization`) for deployment on resource-constrained devices.

**Expected Output**:
- Reduced loss on the fine-tuning dataset.
- Improved performance on subtopic-specific prompts.

**Best Practices**:
- Use a small, high-quality dataset for fine-tuning to avoid overfitting.
- Save separate checkpoints for fine-tuned models.
- Test fine-tuned model on subtopic-specific prompts.

## ✅ Step 8: Inference API or CLI

**Goal**: Create a user-friendly interface to query EEEGPT.

**What You’ll Do**:
- Build a Streamlit web app for interactive querying.
- Alternatively, create a command-line interface (CLI).

**Code** (Streamlit App):
```bash
pip install streamlit
```

```python
import streamlit as st
import torch
from tokenizers import Tokenizer

st.title("EEEGPT: Ask Electrical Engineering Questions")
prompt = st.text_input("Enter your prompt:", "In electrical engineering, a transformer is")
if st.button("Generate"):
    tokenizer = Tokenizer.from_file("data/processed/tokenizer.json")
    model = NanoGPT(vocab_size=5000, d_model=128, n_layers=2, n_heads=4, d_ff=512, max_seq_len=256)
    model.load_state_dict(torch.load('models/model_epoch10.pth'))
    model.to(device).eval()
    generated = generate_text(model, tokenizer, prompt, max_len=50)
    st.write(f"Generated: {generated}")
```

**Explanation**:
- **Streamlit App**: Provides a web interface where users input prompts and view generated text.
- **CLI Alternative**: A simple script to input prompts via the terminal (not shown for brevity).
- **Model Loading**: Load the trained model and tokenizer for inference.

**Expected Output**:
- A web interface at `http://localhost:8501` (run `streamlit run script.py`).
- Generated text based on user prompts.

**Best Practices**:
- Ensure the model and tokenizer paths are correct.
- Handle long prompts by truncating to `max_seq_len`.
- Test the interface with diverse EEE questions.

## ✅ Step 9: Model Export and Use

**Goal**: Export EEEGPT for deployment on edge devices (e.g., Raspberry Pi).

**What You’ll Do**:
- Export the model to TorchScript.
- Provide instructions for loading in other applications.

**Code**:
```python
# Export to TorchScript
model = NanoGPT(vocab_size=5000, d_model=128, n_layers=2, n_heads=4, d_ff=512, max_seq_len=256)
model.load_state_dict(torch.load('models/model_epoch10.pth'))
model.eval()
traced_model = torch.jit.trace(model, torch.randint(0, 5000, (1, 256)).to(device))
traced_model.save('models/eeegpt.pt')

# Example: Load in Python
loaded_model = torch.jit.load('models/eeegpt.pt')
```

**Explanation**:
- **TorchScript**: Converts the model to a format suitable for C++ or mobile deployment.
- **Deployment**: The exported model can be loaded on devices like Raspberry Pi using PyTorch’s C++ API or Python.
- **ONNX Alternative**: Use `torch.onnx.export` for broader compatibility (not shown).

**Expected Output**:
- `eeegpt.pt` file in `models/` for deployment.
- Successful loading and inference in target environments.

**Best Practices**:
- Test the exported model to ensure it produces identical outputs.
- Optimize for edge devices using quantization if needed.
- Document input/output formats for integration.

## Additional Notes
- **Dataset Expansion**: If open-source data is limited, generate synthetic Q&A using templates or existing datasets like [Electrical-engineering](https://huggingface.co/datasets/STEM-AI-mtl/Electrical-engineering).
- **Performance**: Expect moderate performance due to the small model size; fine-tuning can improve domain-specific accuracy.
- **Resources**: Refer to [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) for additional code examples and