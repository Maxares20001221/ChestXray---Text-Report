import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
import torch
from torch.nn.functional import softmax
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm

# Generate the result from validation dataset and compare to the ground truth
# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load the latest checkpoint
ckpt_path = "/content/drive/MyDrive/Colab Notebooks/HC Assignments/Final Project/CKP/model_epoch5.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model = model.to(device)
model.eval()

# Top-k sampling generation
def generate_with_sampling(model, img_embedding, tokenizer, max_len=128, top_k=50, temperature=1.0):
    input_ids = None
    generated = []

    for _ in range(max_len):
        if input_ids is None:
            inputs = img_embedding
        else:
            text_embed = model.gpt.transformer.wte(input_ids)
            inputs = torch.cat([img_embedding, text_embed], dim=1)

        with torch.no_grad():
            logits = model.gpt(inputs_embeds=inputs).logits[:, -1, :]
            logits = logits / temperature
            probs = softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, top_k)
            next_token = topk_indices[0, torch.multinomial(topk_probs[0], 1)].unsqueeze(0)

        if next_token.item() == tokenizer.eos_token_id:
            break

        generated.append(next_token.item())
        input_ids = next_token if input_ids is None else torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True)

# Loop through 10 samples
for idx in range(10):
    sample = val_dataset[idx]
    image = sample["image"].unsqueeze(0).to(device)
    gt_report = sample["caption"]

    # Encode image
    with torch.no_grad():
        feats = model.cnn(image)
        feats = feats.view(1, -1)
        img_embedding = model.img_proj(feats).unsqueeze(1)

    # Generate report using Top-k Sampling
    gen_report = generate_with_sampling(model, img_embedding, tokenizer, max_len=128, top_k=50, temperature=1.0)

    # Show image
    plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu() * 0.5 + 0.5)
    plt.axis('off')
    plt.title(f"Sample {idx+1} - Chest X-ray")
    plt.show()

    print(f"Sample {idx+1}")
    print("Ground Truth Report:\n", gt_report)
    print("\nGenerated Report:\n", gen_report)
    print("-" * 80)


# Evaluate the model performance using BLEU METEOR and ROUGE
generated_reports = []
ground_truth_reports = []

print("🔍 Generating reports for evaluation...")
for idx in tqdm(range(len(val_dataset)), desc="Generating Samples"):
    sample = val_dataset[idx]
    image = sample["image"].unsqueeze(0).to(device)
    gt_report = sample["caption"]

    with torch.no_grad():
        feats = model.cnn(image)
        feats = feats.view(1, -1)
        img_embedding = model.img_proj(feats).unsqueeze(1)

    gen_report = generate_with_sampling(model, img_embedding, tokenizer)
    ground_truth_reports.append(gt_report)
    generated_reports.append(gen_report)

bleu1_scores = []
bleu4_scores = []
meteor_scores = []
rouge_l_scores = []

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1

print("📊 Evaluating text similarity metrics...")
for ref, hyp in tqdm(zip(ground_truth_reports, generated_reports), total=len(val_dataset), desc="Evaluating"):
    ref_tokens = nltk.word_tokenize(ref.lower())
    hyp_tokens = nltk.word_tokenize(hyp.lower())

    bleu1_scores.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth))
    bleu4_scores.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
    meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))
    rouge_l_scores.append(scorer.score(ref, hyp)["rougeL"].fmeasure)

print(f"\n✅ Final Evaluation on {len(val_dataset)} Validation Samples")
print(f"BLEU-1 Score:   {sum(bleu1_scores) / len(bleu1_scores):.4f}")
print(f"BLEU-4 Score:   {sum(bleu4_scores) / len(bleu4_scores):.4f}")
print(f"METEOR Score:   {sum(meteor_scores) / len(meteor_scores):.4f}")
print(f"ROUGE-L Score:  {sum(rouge_l_scores) / len(rouge_l_scores):.4f}")