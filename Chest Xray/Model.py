# Model Architecture (ResNet50 + GPT-2)
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CXRReportGenerator(nn.Module):
    def __init__(self, gpt_model_name="gpt2", image_feat_dim=768):
        super().__init__()

        # Load pretrained GPT2 model
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name)

        # Optionally unfreeze some GPT2 layers for fine-tuning
        for name, param in self.gpt.named_parameters():
            if any(n in name for n in ["ln_f", "transformer.h.10", "transformer.h.11"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Load pretrained ResNet50 backbone, remove final layers
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, 7, 7]

        # Project ResNet output to GPT2 embedding size (with normalization & dropout)
        self.img_proj = nn.Sequential(
            nn.Linear(2048 * 7 * 7, image_feat_dim),
            nn.LayerNorm(image_feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Optional positional embedding for the image token
        self.img_pos = nn.Parameter(torch.randn(1, 1, image_feat_dim))

        # Tokenizer setup (make sure pad_token is defined)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, images, input_ids=None, labels=None):
        B = images.size(0)

        # CNN encode the image
        with torch.no_grad():  # Freeze ResNet
            feats = self.cnn(images)  # [B, 2048, 7, 7]
        feats = feats.view(B, -1)    # [B, 2048*7*7]
        img_token = self.img_proj(feats).unsqueeze(1)  # [B, 1, 768]
        img_token = img_token + self.img_pos           # Add position info

        # GPT2 text embedding
        if input_ids is not None:
            txt_embeds = self.gpt.transformer.wte(input_ids)  # [B, L, 768]
            gpt_input = torch.cat([img_token, txt_embeds], dim=1)  # [B, L+1, 768]
        else:
            gpt_input = img_token  # for generation

        # Add dummy label at the image token position
        if labels is not None:
            dummy = torch.full((labels.size(0), 1), -100).to(labels.device)
            labels = torch.cat([dummy, labels], dim=1)

        outputs = self.gpt(inputs_embeds=gpt_input, labels=labels)
        return outputs.loss, outputs.logits

# Instantiate the model
model = CXRReportGenerator().to(device)
print(model)