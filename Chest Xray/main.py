import os
from DataLoader import train_dataset, val_dataset
from Model import CXRReportGenerator
from Training import train_model
from Visualization import plot_loss_curve
from Validation_Evaluation import evaluate_model, generate_examples

def main():
    # === Step 1: Initialize model ===
    print("🚀 Initializing model...")
    model = CXRReportGenerator()

    # === Step 2: Train model ===
    print("\n📚 Starting training...")
    train_model(model, train_dataset, val_dataset, epochs=20)

    # === Step 3: Visualize loss ===
    print("\n📈 Plotting training/validation loss...")
    plot_loss_curve()

    # === Step 4: Generate inference samples ===
    print("\n🧠 Running inference on validation set...")
    generate_examples(model, val_dataset)

    # === Step 5: Evaluate metrics ===
    print("\n📊 Evaluating generated reports...")
    evaluate_model(model, val_dataset)

if __name__ == "__main__":
    main()
