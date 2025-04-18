# Loss Curve Visualization
import pandas as pd
import matplotlib.pyplot as plt

# Create train loss and validation loss based on the log
history = {
    "epoch": list(range(1, 21)),
    "train_loss": [
        1.8852, 1.0812, 0.9515, 0.8788, 0.8303,
        0.7901, 0.7579, 0.7309, 0.7079, 0.6832,
        0.6652, 0.6472, 0.6265, 0.6113, 0.5941,
        0.5843, 0.5697, 0.5531, 0.5419, 0.5281
    ],
    "val_loss": [
        1.0348, 0.8704, 0.8011, 0.7489, 0.7202,
        0.6958, 0.6807, 0.6644, 0.6495, 0.6420,
        0.6401, 0.6231, 0.6198, 0.6168, 0.6125,
        0.6129, 0.6073, 0.6065, 0.6069, 0.6040
    ]
}

df = pd.DataFrame(history)
df.to_csv(".../loss_log_manual.csv", index=False)

# Plot the loss curve with epochs
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["train_loss"], marker='o', label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], marker='o', label="Val Loss")
plt.title("Training and Validation Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()