from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from transformers import GPT2Tokenizer

# Config
MAX_LEN = 128
BATCH_SIZE = 16
TEST_SIZE = 0.2
SEED = 42

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Custom Dataset class
class CXRReportDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=MAX_LEN):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, report = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        encoded = self.tokenizer(
            report,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        return {
            "image": image,
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "caption": report
        }

# Return dataset & dataloaders
def get_dataloaders(data_pairs, batch_size=BATCH_SIZE):
    train_pairs, val_pairs = train_test_split(data_pairs, test_size=TEST_SIZE, random_state=SEED)

    train_dataset = CXRReportDataset(train_pairs, tokenizer)
    val_dataset   = CXRReportDataset(val_pairs, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader

# Optional debug/test block
if __name__ == "__main__":
    # Make sure `data_pairs` is defined elsewhere or loaded in for testing
    from some_module import data_pairs  # ← 替换为实际路径
    train_dataset, val_dataset, train_loader, val_loader = get_dataloaders(data_pairs)

    sample_batch = next(iter(val_loader))
    print("Image shape:", sample_batch["image"].shape)
    print("Input IDs shape:", sample_batch["input_ids"].shape)
    print("Sample report:", sample_batch["caption"][0])
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))
