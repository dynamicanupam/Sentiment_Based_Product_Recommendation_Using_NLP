# Sentiment Based Product Recommendation System
---
## Problem Statement

E-commerce businesses have transformed the way consumers shop, offering convenience and a wide range of choices. Companies like Amazon, Flipkart, and Myntra have set industry standards by providing personalized shopping experiences.  

Ebuss, a growing e-commerce company, operates across multiple product categories, including household essentials, books, personal care products, medicines, cosmetics, electrical appliances, kitchenware, and health products. To compete with established market leaders, Ebuss aims to enhance its product recommendation system by leveraging user feedback such as reviews and ratings.  

The goal is to develop a sentiment-based product recommendation system to improve user experience and drive customer satisfaction. This involves the following key tasks:  

1. **Data Sourcing and Sentiment Analysis**: Collecting and analyzing user reviews to extract meaningful sentiment insights.  
2. **Building a Recommendation System**: Creating a recommendation engine based on user preferences and interactions.  
3. **Improving Recommendations**: Integrating sentiment analysis results to refine and personalize product suggestions.  
4. **End-to-End Deployment**: Implementing the solution with a user-friendly interface for seamless customer interaction.  

This initiative will empower Ebuss to deliver a superior and personalized shopping experience, positioning it as a strong competitor in the e-commerce industry.

## Data sourcing and sentiment analysis
To analyze product reviews by applying text preprocessing steps and building an ML model to determine the sentiments associated with users' reviews and ratings for various products.

The dataset for this task is a subset derived from a Kaggle competition dataset, tailored specifically for this purpose and provided below.

## Solution Approach
- **Data Preparation**: The dataset and its attribute descriptions are located in the dataset folder. Data cleaning, visualization, and preprocessing are performed using NLP techniques.  

- **Text Vectorization**: The textual data (combination of `review_title` and `review_text`) is vectorized using the TF-IDF Vectorizer, which quantifies the importance of words relative to the entire dataset. 

- **Addressing Class Imbalance**: To tackle class imbalance in the dataset, the Synthetic Minority Over-sampling Technique (SMOTE) is applied for oversampling before model training.  

- **Machine Learning Models**: Multiple classification models are trained on the vectorized data to predict the sentiment (`user_sentiment`) as positive (1) or negative (0). These models include Logistic Regression, and tree-based algorithms like Random Forest, XGBoost and LightGBM. The best-performing model is selected based on evaluation metrics such as Accuracy, Precision, Recall, F1 Score, and AUC. XGBoost emerges as the best model.  

- **Recommender System**: A collaborative filtering-based recommender system is implemented using both user-user and item-item approaches. The system is evaluated using the Root Mean Square Error (RMSE) metric.  

- **Codebase**: The entire implementation for sentiment classification and the recommender system is consolidated in the `Main.ipynb` Jupyter notebook.  

- **Product Filtering**: The recommender system identifies the top 20 products. For these products, the `user_sentiment` is predicted for all reviews, and the 5 products with the highest positive sentiment are highlighted.  

## Demo of the application
[![Watch the video](https://raw.githubusercontent.com/dynamicanupam/Sentiment_Based_Product_Recommendation_Using_NLP/main/Recommendation_App_UI.png)](https://raw.githubusercontent.com/dynamicanupam/Sentiment_Based_Product_Recommendation_Using_NLP/main/Demo-App.mp4)

