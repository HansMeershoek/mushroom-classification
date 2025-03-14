#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mushroom Classification Script
This script downloads the UCI Mushroom Dataset, processes it, trains a Random Forest classifier,
and saves predictions to a CSV file.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define constants
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
DATA_DIR = "mushroom_data"
DATA_FILE = os.path.join(DATA_DIR, "mushroom.data")
TRAIN_FILE = os.path.join(DATA_DIR, "train_data.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_data.csv")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "mushroom_predictions.csv")

# Column names for the dataset
COLUMN_NAMES = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat"
]

def download_and_prepare_data():
    """
    Download the UCI Mushroom Dataset and prepare it for training.
    Returns:
        X_train, X_test, y_train, y_test: Training and testing datasets
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete!")
    
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
    
    # Save training and testing datasets
    train_df = pd.concat([y_train, X_train], axis=1)
    test_df = pd.concat([y_test, X_test], axis=1)
    
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    
    print(f"Training data saved to {TRAIN_FILE}")
    print(f"Testing data saved to {TEST_FILE}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a Random Forest classifier on the training data.
    Args:
        X_train: Training features
        y_train: Training labels
    Returns:
        best_model: The trained model with best parameters
    """
    print("Training model...")
    
    # Create a pipeline with preprocessing and classifier
    pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define hyperparameters for grid search
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Edible", "Poisonous"]))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Edible", "Poisonous"], rotation=45)
    plt.yticks(tick_marks, ["Edible", "Poisonous"])
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the confusion matrix plot
    plt.savefig(os.path.join(DATA_DIR, "confusion_matrix.png"))
    print(f"Confusion matrix saved to {os.path.join(DATA_DIR, 'confusion_matrix.png')}")
    
    return y_pred

def save_predictions(X_test, y_test, y_pred):
    """
    Save the test data and predictions to a CSV file.
    Args:
        X_test: Test features
        y_test: True labels
        y_pred: Predicted labels
    """
    # Create a DataFrame with test data and predictions
    test_df = pd.concat([pd.Series(y_test).reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)
    test_df["predicted"] = y_pred
    test_df["correct"] = test_df["class"] == test_df["predicted"]
    
    # Save to CSV
    test_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Predictions saved to {PREDICTIONS_FILE}")

def main():
    """Main function to run the mushroom classification pipeline."""
    print("Starting Mushroom Classification...")
    
    # Download and prepare data
    X_train, X_test, y_train, y_test = download_and_prepare_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Save predictions
    save_predictions(X_test, y_test, y_pred)
    
    print("Mushroom Classification completed successfully!")

if __name__ == "__main__":
    main()