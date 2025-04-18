# 🩻 Chest X-ray Report Generator (ResNet50 + GPT-2)

This project builds an end-to-end deep learning pipeline to generate **radiology reports from chest X-ray images**, combining **ResNet50 for image encoding** and **GPT-2 for text decoding**.

---

## 🚀 Model Architecture

- **Encoder**: Pretrained ResNet-50 (removes classifier head, extracts 2048×7×7 features)
- **Projector**: Linear layer → ReLU → LayerNorm to map to GPT2 hidden size (768)
- **Decoder**: GPT2LMHeadModel (pretrained), with top-k sampling generation
- **Loss**: CrossEntropy with ignore index `-100`, trained using teacher forcing

---

## 📊 Dataset & Preprocessing

- **Dataset**: Indiana University Chest X-ray Dataset
- **Preprocessing**:
  - Resize → Grayscale → Normalize
  - Text tokenized via GPT2 tokenizer (pad token = EOS)

---

## 📈 Training Results

- ✅ **Final Training Loss**: `0.5281`
- ✅ **Final Validation Loss**: `0.6040`
- ✅ **Validation Evaluation on 766 Samples**:
  - **BLEU-1**: `0.3140`
  - **BLEU-4**: `0.0811`
  - **METEOR**: `0.3034`
  - **ROUGE-L**: `0.2569`

---

## 🧪 Inference Example

Generated reports use **Top-k sampling (k=50)** with temperature 1.0.

