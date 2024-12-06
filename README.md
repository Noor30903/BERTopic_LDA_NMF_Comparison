# Product Description Classification Using Topic Modeling Models: NMF, LDA, BERTopic 

This project explores the application of three advanced machine learning techniques—**Non-negative Matrix Factorization (NMF)**, **Latent Dirichlet Allocation (LDA)**, and **BERTopic**—to classify product descriptions into predefined categories. The goal of this project is to automatically categorize product descriptions and map them to specific zones based on their content, improving the organization and management of products in inventory systems.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)

---

## Overview

This project combines three different topic modeling approaches: **NMF**, **LDA**, and **BERTopic**. All three techniques are used to classify product descriptions into categories like "beauty," "books," "electronics," etc., and map these categories to their respective zones (e.g., "Cosmetic Zone," "Dry Zone"). The project consists of the following stages:

1. **Data Preprocessing**: Text cleaning and preparation using NLP techniques such as lemmatization and stopword removal.
2. **NMF Model**: A topic modeling approach used to extract topics from product descriptions, which are then classified using a Logistic Regression model.
3. **LDA Model**: Another topic modeling approach, Latent Dirichlet Allocation, which assigns topics to documents and then classifies them using machine learning.
4. **BERTopic Model**: A state-of-the-art topic modeling technique using embeddings from SentenceTransformer and clustering with KMeans.
5. **Model Evaluation**: The models are evaluated based on accuracy, precision, recall, F1-score, and coherence score.

---

## Features

- Topic modeling with **NMF**, **LDA**, and **BERTopic** for automated text classification.
- **Logistic Regression** for classifying the topics into product categories.
- **Coherence Score** to evaluate the interpretability of the extracted topics.
- Mapping of predicted categories to product zones (e.g., "Cosmetic Zone" for beauty-related products).
- High-performance classification with **BERTopic** achieving 97% accuracy.
- Easy-to-use preprocessing pipeline for text cleaning and transformation.
- Comparison between NMF, LDA, and BERTopic models for topic modeling effectiveness.

---


## Dataset Information
We used a Big Amazon products dataset from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) containing 8.5 million products with metadata like category, title, description, and price. For this project, we focused on the product description for topic modeling and the main category as the target. From five categories (home, beauty, electronics, food, and books), we randomly sampled 268,753 products, balanced with 50,000 products per category except for beauty, which included 68,753 products.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo-name.git
    cd your-repo-name
    ```

2. Download the required datasets and place them in the `dataset/` directory.

3. Make sure you have a Python environment with the necessary libraries installed.

---

## Usage

1. **Data Preprocessing**:
    - Clean and preprocess the text data using NLTK (e.g., tokenization, stopword removal, lemmatization).
    - Load the training and test data into pandas DataFrames.

2. **Train NMF Model**:
    - Fit the NMF model on the training data:
    ```python
    nmf.fit(X_train, y_train)
    ```

3. **Train LDA Model**:
    - Preprocess the data and train the LDA model for topic extraction:
    ```python
    lda_model = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=5)
    ```
    - Extract and display topics:
    ```python
    topics = lda_model.print_topics(num_words=10)
    for topic in topics:
        print(topic)
    ```

4. **Train BERTopic Model**:
    - Fit the BERTopic model on the product descriptions:
    ```python
    topic_model.fit(documents)
    ```

5. **Model Evaluation**:
    - Evaluate the models on the test data:
    ```python
    accuracy = accuracy_score(y_test, predicted_categories)
    print(classification_report(y_test, predicted_categories))
    ```

6. **Mapping to Zones**:
    - Map the predicted product categories to their respective zones:
    ```python
    predicted_zones = [category_to_zone.get(category, "unknown zone") for category in predicted_categories]
    ```

---

## Results

### 3.Accuracy:
- **NMF**: 72%
- **LDA**:  75%
- **BERTopic**: 97%

### 4. Coherence Score:
- **NMF**: 0.72 (indicative of meaningful topics)
- **LDA**: 0.45 (lower coherence score but effective in classification)
- **BERTopic**: 0.71 (demonstrates well-structured topics)

---

## Contributors

- **Noor Alawlaqi** - S21107270
- **Maha Almashharawi** - S20106480
- **Mashael Alsalamah** - S20206926

