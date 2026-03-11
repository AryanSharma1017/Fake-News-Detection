# Fake News Detection using Transformer Encoder

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![NLP](https://img.shields.io/badge/NLP-Transformer-orange)
![Framework](https://img.shields.io/badge/Framework-Streamlit-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)
![Model](https://img.shields.io/badge/Model-Transformer-purple)

A deep learning project that detects **fake vs real news articles** using a **Transformer-based text classification model built from scratch in PyTorch**.

Unlike many NLP projects that rely on prebuilt architectures, this implementation constructs the **core Transformer components manually**, including:

* Token + Positional Embeddings
* Multi-head self-attention
* Scaled dot-product attention
* Feedforward neural networks
* Encoder stacks
* A classification head for binary prediction

The model learns contextual relationships between words in a news article to determine whether the article is **real or fake**.

---

# 🚀 Project Overview

Fake news spreads rapidly across social media and digital platforms. Traditional NLP methods struggle to capture **long-range dependencies between words** in articles.

This project solves that problem using the **Transformer architecture**, which allows every token in a sequence to attend to every other token.

The system processes news text and predicts whether the article belongs to one of two classes:

* **0 → Real News**
* **1 → Fake News**

---

# 🧠 Model Architecture

The model follows a **Transformer Encoder classification pipeline**:

```
Raw Text
   ↓
Tokenization
   ↓
Vocabulary Encoding
   ↓
Token + Positional Embedding
   ↓
Transformer Encoder Blocks (Self Attention)
   ↓
Sentence Representation
   ↓
Feedforward Classification Head
   ↓
Fake / Real Prediction
```

---

# 🔬 Key Components

## 1️⃣ Tokenization

The text is cleaned and tokenized using whitespace-based tokenization.

```
"The economy is growing fast"
→ ["The", "economy", "is", "growing", "fast"]
```

---

## 2️⃣ Vocabulary Construction

A vocabulary is built from the dataset using frequency statistics.

Special tokens:

| Token   | Purpose                   |
| ------- | ------------------------- |
| `<PAD>` | Padding shorter sequences |
| `<UNK>` | Unknown words             |

Parameters used:

```
max_vocab_size = 10,000
min_frequency = 2
```

---

## 3️⃣ Text Encoding

Each article is converted into a **fixed-length sequence**.

```
Max sequence length = 512 tokens
```

Shorter texts are padded.

---

## 4️⃣ Token + Positional Embeddings

Transformers cannot understand word order by default.

Therefore the model combines:

```
Token Embedding
+
Positional Embedding
```

This gives the model information about:

* word meaning
* word position

Output shape:

```
[batch_size, sequence_length, embedding_dimension]
```

---

# ⚡ Scaled Dot Product Attention

The core of the transformer.

The model computes attention using:

```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) V
```

Where:

| Symbol | Meaning                  |
| ------ | ------------------------ |
| Q      | Query                    |
| K      | Key                      |
| V      | Value                    |
| d_k    | dimension of key vectors |

This allows each word to **focus on relevant words across the sentence**.

---

# 🧩 Multi-Head Attention

Instead of using a single attention mechanism, the model splits attention into multiple heads.

Example:

```
embed_dim = 128
num_heads = 4
head_dim = 32
```

Each head learns **different linguistic relationships** such as:

* grammar
* context
* topic relevance
* semantic similarity

---

# 🧱 Transformer Encoder Block

Each encoder block consists of:

1️⃣ Multi-head self attention
2️⃣ Residual connection
3️⃣ Layer normalization
4️⃣ Feedforward neural network

Structure:

```
Input
 ↓
Self Attention
 ↓
Add & Normalize
 ↓
Feedforward Network
 ↓
Add & Normalize
 ↓
Output
```

Multiple encoder blocks are stacked to build deeper representations.

---

# 🎯 Classification Head

After the transformer encoding stage, a sentence representation is extracted and passed through a classifier.

Architecture:

```
Linear Layer
ReLU Activation
Dropout
Linear Output Layer
```

Final output:

```
[batch_size, 2]
```

representing probabilities for:

* Real News
* Fake News

---

# 📊 Training Configuration

| Parameter               | Value            |
| ----------------------- | ---------------- |
| Embedding Dimension     | 128              |
| Number of Heads         | 4                |
| Encoder Layers          | 2                |
| Feedforward Hidden Size | 256              |
| Max Sequence Length     | 512              |
| Batch Size              | 32               |
| Epochs                  | 5                |
| Optimizer               | Adam             |
| Learning Rate           | 0.001            |
| Loss Function           | CrossEntropyLoss |

---

# 📈 Training Process

Training follows the standard deep learning workflow:

1. Load dataset
2. Build vocabulary
3. Encode text sequences
4. Create PyTorch datasets
5. Train transformer model
6. Evaluate predictions

Training loop performs:

```
Forward pass
Loss calculation
Backward propagation
Optimizer step
Accuracy calculation
```

---

# 🧪 Dataset Pipeline

Custom PyTorch dataset:

```
NewsDataset
```

DataLoader handles:

* batching
* shuffling
* efficient GPU training

---


# 📤 Model Saving

The trained model is saved using PyTorch:

```
torch.save(model.state_dict(), "transformer_model.pth")
```

---

# 🖥️ Deployment

The trained model can be deployed using **Streamlit** to build an interactive web application where users can paste news text and receive predictions instantly.

---

# 📊 Example Prediction

Input:

```
Breaking: Government confirms new economic policy changes today
```

Output:

```
Prediction: Real News
Confidence: 92%
```

---

# 🧠 Future Improvements

Possible enhancements:

* Add pretrained embeddings (GloVe / FastText)
* Add CLS token representation
* Increase encoder depth
* Use larger datasets
* Integrate pretrained transformer models

---

# 🛠️ Technologies Used

* Python
* PyTorch
* NumPy
* Scikit-Learn
* Matplotlib
* Streamlit

---

# 👨‍💻 Author

**Aryan Sharma**

Software Engineering Student
AI & Machine Learning Enthusiast



---

# ⭐ Contributing

Pull requests and suggestions are welcome.

---

# 📜 License

MIT License
