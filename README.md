# Naive Bayes Spam Classification

This project implements a spam classification model using the Naive Bayes algorithm and a Support Vector Machine (SVM). It processes SMS messages, classifies them as spam or not spam, and evaluates the performance of both classifiers.

## Overview

The main components of this project include:
- Preprocessing of SMS messages
- Creation of a word dictionary
- Transformation of messages into numerical matrices
- Implementation of the Naive Bayes classification algorithm
- Implementation of the Support Vector Machine (SVM) classification algorithm
- Evaluation of model accuracy and performance

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Any additional libraries as specified in the requirements

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/ChewieMewK/naive-bayes-spam-classification.git
cd naive-bayes-spam-classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Data

The project requires a dataset of SMS messages for training, validation, and testing. You can use the `spam_train.tsv`, `spam_val.tsv`, and `spam_test.tsv` files provided in the `data` directory.

### Usage

To run the spam classification model, execute the main script:

```bash
python main.py
```

### Results

The script will output the accuracy of the Naive Bayes classifier on the testing set and display the top 5 indicative words for identifying spam. Additionally, it will compute the optimal SVM radius and report the accuracy of the SVM model.

## Functions

### Key Functions

- **get_words(message)**: Normalize and split SMS messages into words.
- **create_dictionary(messages)**: Create a dictionary mapping words to integer indices, filtering out rare words.
- **transform_text(messages, word_dictionary)**: Convert SMS messages into a numerical matrix of word counts.
- **fit_naive_bayes_model(matrix, labels)**: Fit a Naive Bayes model to the training data.
- **predict_from_naive_bayes_model(model, matrix)**: Make predictions using the trained Naive Bayes model.
- **compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider)**: Determine the optimal SVM radius based on accuracy.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## Acknowledgments

- This project is inspired by the Naive Bayes and SVM algorithms covered in various machine learning courses and literature.
```
