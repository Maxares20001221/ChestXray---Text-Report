## 🌐 Project Overview
This repository contains an end-to-end pipeline for generating radiology reports from chest X-ray images. The model leverages a ResNet50 image encoder and a GPT-2 text decoder to generate descriptive findings directly from medical images.

Dataset: **Indiana University Chest X-ray Collection (IU-CXR)**

---

## 🚀 How to Run (via `main.py`)
```bash
# Step 1: Clone the repository
$ git clone https://github.com/yourusername/Chest-Xray-Report-Generation.git
$ cd Chest-Xray-Report-Generation

# Step 2: Install dependencies
$ pip install -r requirements.txt

# Step 3: Ensure the IU-CXR .tgz files are placed correctly
# (Edit paths in `DataLoading.py` if needed)

# Step 4: Run the pipeline end-to-end
$ python main.py
```

---

## 📒 Notebook Version Available
For step-by-step visualization and experimentation, you can run:

- `ChestXray_Report_Generation.ipynb`

This notebook includes training, visualization, and inference outputs.

---

## 🔮 Directory Structure
```
Chest-Xray-Report-Generation/
├── DataLoading.py              # Extract & parse IU-CXR images and reports
├── DataLoader.py               # Custom PyTorch Dataset & DataLoader
├── Model.py                    # CXRReportGenerator model (ResNet50 + GPT-2)
├── Training.py                 # Train model and save checkpoints
├── Validation_Evaluation.py    # Inference and metric evaluation
├── Visualization.py            # Plotting training curves
├── main.py                     # Run full pipeline
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## 📉 Evaluation Results
Performance on 766 validation samples:

| Metric      | Score  |
|-------------|--------|
| BLEU-1      | 0.3140 |
| BLEU-4      | 0.0811 |
| METEOR      | 0.3034 |
| ROUGE-L     | 0.2569 |

---

## 🔧 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
Or manually ensure:
```
torch
transformers
scikit-learn
nltk
rouge-score
matplotlib
pillow
tqdm
```

---

## 📊 Inference Example
```python
image = val_dataset[0]['image'].unsqueeze(0).to(device)
feats = model.cnn(image).view(1, -1)
embedding = model.img_proj(feats).unsqueeze(1)
report = generate_with_sampling(model, embedding, tokenizer)
```
Output:
```
Ground Truth:  Heart mildly enlarged. No effusion.
Generated:     Heart size is mildly enlarged. No pleural effusion.
```

---

## 🔬 Future Work
- Replace ResNet50 with ViT or Swin Transformer
- Integrate beam search and coverage mechanisms
- Fine-tune on MIMIC-CXR or larger clinical datasets
- Add sectioned report generation (e.g., Findings / Impressions separately)

---

## ✨ Acknowledgements
- Indiana University CXR dataset
- HuggingFace Transformers
- torchvision pretrained ResNet

---

