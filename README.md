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
Manually ensure:
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
Ground Truth: The heart is near top normal in size with tortuosity of the aorta. The pulmonary vascular markings are symmetric and normal. There are low lung volumes with XXXX opacities consistent with focal atelectasis. There is no pleural effusion or pneumothorax. There are degenerative changes in thoracic spine and thoracic kyphosis. Low lung volumes with XXXX opacities consistent with focal atelectasis.
Generated Report: The lungs are hyperexpanded. There is no pleural effusion or pneumothorax. Cardiomediastinal silhouette is within normal limits. Lungs are hyperexpanded with no acute abnormality identified. No acute pulmonary abnormality identified.
--------------------------------------------------------------------------------
```
---

## Future Work
To further improve the quality and clinical utility of chest X-ray report generation, the following directions are proposed:

Structured Report Generation
Introduce hierarchical decoders or section-aware mechanisms that explicitly generate structured sections such as Impression, Findings, and Observation for better clinical readability and alignment with radiology standards.

Multi-View Learning
Incorporate both frontal and lateral chest X-ray images to enable the model to capture spatial relationships and improve the detection of abnormalities not visible in a single view.

Domain-Specific Language Models
Replace the GPT-2 decoder with a biomedical-oriented language model like BioGPT, PubMedGPT, or a ClinicalBERT decoder, to better handle specialized terminology and sentence patterns common in medical reporting.

Reinforcement Learning from Human Feedback (RLHF)
Apply RLHF using clinician-provided feedback or report quality scores to directly optimize for clinical accuracy, fluency, and completeness.

Retrieval-Augmented Generation (RAG)
Enhance generation by retrieving similar historical X-ray cases or reports as auxiliary input, improving factual grounding and diversity in generated outputs.

---

## ✨ Acknowledgements
- Indiana University CXR dataset
- HuggingFace Transformers
- torchvision pretrained ResNet

---

