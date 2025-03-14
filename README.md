# Mushroom Classification

A machine learning project for classifying mushrooms as edible or poisonous using a Random Forest classifier.

## Project Overview

This project uses the UCI Mushroom Dataset to build a machine learning model that can predict whether a mushroom is edible or poisonous based on its physical characteristics. The model achieves high accuracy and recall, making it reliable for mushroom classification.

## Dataset

The dataset used is the [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom), which includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended (the latter class was combined with the poisonous one).

The dataset contains 8,124 mushroom samples with 22 categorical features such as:
- Cap shape, surface, and color
- Bruises presence
- Odor
- Gill attachment, spacing, size, and color
- Stalk shape and root
- And many more

## Model

The classification model uses a pipeline with:
1. **OneHotEncoder** for preprocessing categorical features
2. **RandomForestClassifier** for classification

The model's performance was optimized through GridSearchCV, with a primary objective of maximizing recall. This is crucial for mushroom classification, as minimizing false negatives (i.e., misclassifying poisonous mushrooms) is paramount. A key strategy employed was the use of class_weight within the hyperparameter grid to address potential class imbalance. This approach resulted in the model achieving 100% accuracy.

## Results

The model achieves:
- **Accuracy**: 100%
- **Recall**: 100%

This means the model correctly identifies all poisonous mushrooms in the test set, which is crucial for safety.

## Files in the Repository

- `mushroom_classification.py`: The main Python script that downloads the dataset, trains the model, and makes predictions
- `mushroom_data/`: Directory containing the dataset and prediction files
  - `mushroom.data`: The raw UCI Mushroom dataset
  - `train_data.csv`: The training dataset
  - `test_data.csv`: The testing dataset
  - `mushroom_predictions.csv`: Predictions made by the model

## How to Run

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the script:
   ```
   python mushroom_classification.py
   ```

## Future Improvements

- Implement feature importance analysis to identify the most important characteristics for classification
- Explore other machine learning algorithms for comparison
- Create a web application for real-time mushroom classification

## Warning

This model is for educational purposes only. Never rely on a machine learning model to determine if a mushroom is safe to eat. Always consult with a mushroom expert before consuming any wild mushrooms.
