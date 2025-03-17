#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Confusion Matrix Visualizations
This script loads the trained model and test data from the mushroom classification project
and creates two different visualizations of the confusion matrix:
1. Standard confusion matrix with counts
2. Normalized confusion matrix with percentages
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import pickle

# Define constants (same as in mushroom_classification.py)
DATA_DIR = "mushroom_data"
TEST_FILE = os.path.join(DATA_DIR, "test_data.csv")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "mushroom_predictions.csv")

def load_test_data():
    """
    Load the test data from the CSV file.
    Returns:
        X_test: Test features
        y_test: Test labels
    """
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test data file not found: {TEST_FILE}")
    
    # Read the test dataset
    print(f"Loading test data from {TEST_FILE}...")
    test_df = pd.read_csv(TEST_FILE)
    
    # Split features and target
    X_test = test_df.drop("class", axis=1)
    y_test = test_df["class"]
    
    return X_test, y_test

def load_predictions():
    """
    Load the predictions from the CSV file.
    Returns:
        y_test: True labels
        y_pred: Predicted labels
    """
    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_FILE}")
    
    # Read the predictions file
    print(f"Loading predictions from {PREDICTIONS_FILE}...")
    pred_df = pd.read_csv(PREDICTIONS_FILE)
    
    y_test = pred_df["class"]
    y_pred = pred_df["predicted"]
    
    return y_test, y_pred

def create_standard_confusion_matrix(y_test, y_pred):
    """
    Create and save a standard confusion matrix visualization.
    Args:
        y_test: True labels
        y_pred: Predicted labels
    """
    print("Creating standard confusion matrix...")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Counts)', fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Edible", "Poisonous"], rotation=45, fontsize=12)
    plt.yticks(tick_marks, ["Edible", "Poisonous"], fontsize=12)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center", fontsize=14,
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    
    # Add accuracy information
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, 0.01, f"Accuracy: {accuracy:.4f}", ha="center", fontsize=14)
    
    # Save the confusion matrix plot
    output_file = os.path.join(DATA_DIR, "standard_confusion_matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Standard confusion matrix saved to {output_file}")
    plt.close()

def create_normalized_confusion_matrix(y_test, y_pred):
    """
    Create and save a normalized confusion matrix visualization.
    Args:
        y_test: True labels
        y_pred: Predicted labels
    """
    print("Creating normalized confusion matrix...")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot normalized confusion matrix using seaborn for better visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlGnBu',
                xticklabels=["Edible", "Poisonous"],
                yticklabels=["Edible", "Poisonous"])
    
    plt.title('Normalized Confusion Matrix (Percentages)', fontsize=16)
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    
    # Add accuracy information
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, 0.01, f"Accuracy: {accuracy:.4f}", ha="center", fontsize=14)
    
    # Save the normalized confusion matrix plot
    output_file = os.path.join(DATA_DIR, "normalized_confusion_matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Normalized confusion matrix saved to {output_file}")
    plt.close()

def main():
    """Main function to create confusion matrix visualizations."""
    print("Starting Confusion Matrix Visualizations...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
    
    try:
        # Try to load predictions first
        y_test, y_pred = load_predictions()
    except FileNotFoundError:
        # If predictions file doesn't exist, we need to load test data
        # and make predictions using the model
        print("Predictions file not found. Please run mushroom_classification.py first.")
        return
    
    # Create standard confusion matrix
    create_standard_confusion_matrix(y_test, y_pred)
    
    # Create normalized confusion matrix
    create_normalized_confusion_matrix(y_test, y_pred)
    
    print("Confusion Matrix Visualizations completed successfully!")

if __name__ == "__main__":
    main()