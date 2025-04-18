import os
from DataLoading import data_pairs
from DataLoader import get_dataloaders
from Model import get_model
from Training import train_model
from Visualization import plot_loss_curve
from Validation_Evaluation import evaluate_model, generate_examples

def main():
    # === Step 1: Load and split data ===
    print(" Loading and splitting dataset...")
    train_dataset, val_dataset, train_loader, val_loader = get_dataloaders(data_pairs)

    # === Step 2: Initialize model ===
    print(" Initializing model...")
    model = get_model()

    # === Step 3: Train model ===
    print("\n Starting training...")
    train_model(model, train_dataset, val_dataset, epochs=20)

    # === Step 4: Visualize loss ===
    print("\n Plotting training/validation loss...")
    plot_loss_curve()

    # === Step 5: Generate inference samples ===
    print("\n Running inference on validation set...")
    generate_examples(model, val_dataset)

    # === Step 6: Evaluate model ===
    print("\n Evaluating generated reports...")
    evaluate_model(model, val_dataset)

if __name__ == "__main__":
    main()
