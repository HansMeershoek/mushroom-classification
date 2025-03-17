#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Before and After Confusion Matrices
This script generates two confusion matrix visualizations:
1. Before applying Random Forest - using a simple baseline model (Decision Tree)
2. After applying Random Forest - using the optimized Random Forest model
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

# Define constants
DATA_DIR = "mushroom_data"
DATA_FILE = os.path.join(DATA_DIR, "mushroom.data")
BEFORE_CM_FILE = os.path.join(DATA_DIR, "confusion_matrix_before.png")
AFTER_CM_FILE = os.path.join(DATA_DIR, "confusion_matrix_after.png")

# Column names for the dataset
COLUMN_NAMES = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat"
]

def load_data():
    """
    Load the UCI Mushroom Dataset.
    Returns:
        X_train, X_test, y_train, y_test: Training and testing datasets
    """
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}. Please run mushroom_classification.py first.")
    
    # Read the dataset
    print("Reading dataset...")
    df = pd.read_csv(DATA_FILE, header=None, names=COLUMN_NAMES)
    
    # Map class labels: 'e' (edible) -> 0, 'p' (poisonous) -> 1
    df["class"] = df["class"].map({"e": 0, "p": 1})
    
    # Split features and target
    X = df.drop("class", axis=1)
    y = df["class"]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(cm, title, output_file):
    """
    Plot and save a confusion matrix visualization.
    Args:
        cm: Confusion matrix
        title: Title for the plot
        output_file: File path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
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
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_file}")
    plt.close()

def generate_before_matrix(X_train, X_test, y_train, y_test):
    """
    Generate confusion matrix before applying Random Forest (using Decision Tree).
    Args:
        X_train, X_test, y_train, y_test: Training and testing datasets
    """
    print("Generating 'before' confusion matrix using Decision Tree...")
    
    # Create a simple baseline model (Decision Tree)
    baseline_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    # Train the model
    baseline_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = baseline_pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        cm, 
        "Confusion Matrix - Before (Decision Tree)", 
        BEFORE_CM_FILE
    )

def generate_after_matrix(X_train, X_test, y_train, y_test):
    """
    Generate confusion matrix after applying Random Forest.
    Args:
        X_train, X_test, y_train, y_test: Training and testing datasets
    """
    print("Generating 'after' confusion matrix using Random Forest...")
    
    # Create a Random Forest model with optimized parameters
    rf_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        ))
    ])
    
    # Train the model
    rf_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        cm, 
        "Confusion Matrix - After (Random Forest)", 
        AFTER_CM_FILE
    )

def main():
    """Main function to generate before and after confusion matrices."""
    print("Starting confusion matrix comparison generation...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Generate 'before' confusion matrix
    generate_before_matrix(X_train, X_test, y_train, y_test)
    
    # Generate 'after' confusion matrix
    generate_after_matrix(X_train, X_test, y_train, y_test)
    
    print("\nConfusion matrix comparison completed!")
    print(f"Before matrix (Decision Tree): {BEFORE_CM_FILE}")
    print(f"After matrix (Random Forest): {AFTER_CM_FILE}")

if __name__ == "__main__":
    main()