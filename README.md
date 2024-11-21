# Sentiment Based Product Recommendation System
---
## Problem Statement
The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings. 

In order to do this, you planned to build a sentiment-based product recommendation system, which includes the following tasks.
1-Data sourcing and sentiment analysis
2-Building a recommendation system
3-Improving the recommendations using the sentiment analysis model
4-Deploying the end-to-end project with a user interface
 
## Data sourcing and sentiment analysis
In this task, you have to analyse product reviews after some text preprocessing steps and build an ML model to get the sentiments corresponding to the users' reviews and ratings for multiple products. 

The dataset that you are going to use is inspired by this Kaggle competition. We have made a subset of the original dataset, which has been provided below.


## Solution Approach

- **Data Preparation**: The dataset and attribute descriptions are available under the dataset folder. The data is cleaned, visualized, and preprocessed using Natural Language Processing (NLP).

- **Text Vectorization**: The TF-IDF Vectorizer is used to vectorize the textual data (review_title+review_text). This measures the relative importance of a word with respect to other documents.

- **Handling Class Imbalance**: The dataset suffers from a class imbalance issue. The Synthetic Minority Over-sampling Technique (SMOTE) is used for oversampling before applying the model.

- **Machine Learning Models**: Various Machine Learning Classification Models are applied on the vectorized data and the target column (user_sentiment). These models include Logistic Regression, Naive Bayes, and Tree Algorithms (Decision Tree, Random Forest, XGBoost). The objective of these ML models is to classify the sentiment as positive(1) or negative(0). The best model is selected based on various ML classification metrics (Accuracy, Precision, Recall, F1 Score, AUC). XGBoost is selected as the best model based on these evaluation metrics.

- **Recommender System**: A Collaborative Filtering Recommender system is created based on User-user and Item-item approaches. The Root Mean Square Error (RMSE) evaluation metric is used for evaluation.

- **Code**: The code for Sentiment Classification and Recommender Systems is available in the Main.ipynb Jupyter notebook.

- **Product Filtering**: The top 20 products are filtered using the recommender system. For each of these products, the user_sentiment is predicted for all the reviews. The Top 5 products with the highest Positive User Sentiment are then filtered out.


## Demo of the application
[![Watch the video](https://raw.githubusercontent.com/dynamicanupam/Sentiment_Based_Product_Recommendation_Using_NLP/main/Recommendation_App_UI.png)](https://raw.githubusercontent.com/dynamicanupam/Sentiment_Based_Product_Recommendation_Using_NLP/main/Demo-App.mp4)

