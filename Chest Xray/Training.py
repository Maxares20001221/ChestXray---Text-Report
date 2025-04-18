# Training the model for 20 epochs
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd

def train_model(model, train_dataset, val_dataset, epochs=20, batch_size=16, lr=1e-4,
                save_path=".../CKP",
                device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(save_path, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    model = model.to(device)
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")

        for batch in train_bar:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            loss, _ = model(images, input_ids=input_ids, labels=labels)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(train_loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # ------------------- Validation -------------------
        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
        with torch.no_grad():
            for batch in val_bar:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                val_loss, _ = model(images, input_ids=input_ids, labels=labels)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Save Checkpoint
        if epoch % 5 == 0:
            ckpt_path = os.path.join(save_path, f"model_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

        # Save Loss
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch} completed. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save final log
    log_path = os.path.join(save_path, "loss_log.csv")
    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"Loss log saved to {log_path}")
    print("Training complete!")

# Example usage
train_model(model, train_dataset, val_dataset, epochs=20)