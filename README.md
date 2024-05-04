# Identification of Political Ideology


## Overview

This repository contains tools and models for detecting political bias in news articles within the United States context. It includes various natural language processing (NLP) models and techniques for classifying articles into left, center, or right political biases. Additionally, a summarization tool "google/pegasus-multi_news" was used to generate concise summaries of news articles for balanced consumption.


## Key Features

- **NLP Bias Detection Pipeline:** Used classical NLP tools and models along with the transformer model RoBERTa-base for classifying political biases in news articles.
- **Article Summarization Tool:** Input news articles, paste URLs from sources that interest you, or just search for the most recent articles based on keywords. This feature, powered by the Pegasus transformer model, generates summarized versions of the articles, ensuring efficient and balanced news consumption. Please note that this functionality is available exclusively within the Streamlit application (please refer to the "StreamLit_app" folder).
- **Rich Dataset:** Newspaper Bias Dataset (NewB) created by Jerry Wei and an Article-Bias-Prediction dataset created from allsides.com by Ramy Baly. The Article-Bias-Prediction dataset contains a total of 37,554 articles labeled by political bias.
- **Model Architectures:** This project utilizes the RoBERTa (Robustly optimized BERT approach) transformer-based model for sequence classification. The RoBERTa model is fine-tuned to classify news articles into three categories: left, center, or right political biases. It employs tokenization and attention mechanisms to process input data efficiently and achieve accurate classification results and was trained on Bias Dataset (NewB).
