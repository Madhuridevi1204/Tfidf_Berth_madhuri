import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from openpyxl import load_workbook
import base64
import os
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
df1 = pd.read_csv('C:/Users/maadh/OneDrive/Documents/New Auction/training_data.csv')
df2 = pd.read_csv('C:/Users/maadh/OneDrive/Documents/New Auction/testing_data.csv')
st.title('Auction Recommendation System')
st.caption(" ")
st.caption(" ")
# Create select boxes for user input
df1['train'] = df1['tender_type_name'] + ', ' + df1['tender_category_name'] + ', ' + df1['tender_form_contract_name'] + ', ' + df1['tender_location']
df2['train'] = df2['tender_type_name'] + ', ' + df2['tender_category_name'] + ', ' + df2['tender_form_contract_name'] + ', ' + df2['tender_location']

# Combine both datasets
combined_df = pd.concat([df1, df2])

# User_id = st.selectbox('Select User Id:', df1['bidder_id'].unique())
st.subheader('Enter the bidder information')
bidder_location = st.selectbox('Select bidder_location:', df1['tender_location'].unique())
tender_type_name = st.selectbox('Select tender_type_name:', df1['tender_type_name'].unique())
tender_category_name = st.selectbox('Select tender_category_name:', df1['tender_category_name'].unique())
tender_form_contract_name = st.selectbox('Select tender_form_contract_name:', df1['tender_form_contract_name'].unique())

# Create user input
user = [tender_type_name +', '+ tender_category_name +', '+ tender_form_contract_name +', '+ bidder_location]

# Filter the dataset based on bidder_location and active column
match_location = combined_df.loc[(combined_df['tender_location'] == bidder_location) & (combined_df['active'] == 1)]

# Convert the dataset and user input into feature vectors using the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(user + list(match_location['train']))

# Calculate the cosine similarity between the user input and each row in the dataset
cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

# Train a KNN model using the dataset
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(vectors[1:])
st.sidebar.write('<div style="text-align:left;"><h3>Select the no.of Recommendations</h3></div>', unsafe_allow_html=True)

num_recommendations = st.sidebar.slider('', 1, 20, 10)

# Find the top 10 nearest neighbors using collaborative filtering
collaborative_indices = knn_model.kneighbors(vectors[0:1], n_neighbors=10, return_distance=False)[0]
collab_top_10 = match_location.iloc[collaborative_indices, [0, 2, 3, 4,5, -1]].loc[match_location['active'] == 1]
# Get the indices of the top 10 most similar rows using content-based filtering
content_based_indices = cosine_similarities.argsort()[:-11:-1]
content_top_10 = match_location.iloc[content_based_indices, [0, 2, 3, 4, -1]].loc[match_location['active'] == 1]

# Combine the indices from content-based filtering and collaborative filtering
hybrid_indices = np.unique(np.concatenate((content_based_indices, collaborative_indices), axis=None))
recommend_button_clicked = st.button("Recommend Me")

if recommend_button_clicked:
# Create a dataframe of the top 10 most similar rows from Dataset 2
    top_10 = match_location.iloc[hybrid_indices, [0, 2, 3, 4, 5,-1,9,8]].loc[match_location['active'] == 1]
    
    details = []
    for index, row in top_10.iterrows():
        rating = random.randint(2, 5)
        stars = '⭐️' * rating
        detail = {
        'tender_id': row['tender_id'],
        'tender_type_name': row['tender_type_name'],
        'tender_category_name': row['tender_category_name'],
        'tender_form_contract_name': row['tender_form_contract_name'],
        'tender_location': row['tender_location'],
        'bidder_ratings': stars
        }
        details.append(detail)

# Display the details and append an "Apply" button for each segment of data
    for detail in details:
        col1, col2 = st.columns([4, 1])
    
        with col1:
            st.write(f"tender_id: {detail['tender_id']}")
            st.write(f"tender_type_name: {detail['tender_type_name']}")
            st.write(f"tender_category_name: {detail['tender_category_name']}")
            st.write(f"tender_form_contract_name: {detail['tender_form_contract_name']}")
            st.write(f"tender_location: {detail['tender_location']}")
            st.write(f"bidder_ratings: {detail['bidder_ratings']}")
    
        with col2:
            col2.empty()  # Add an empty placeholder
            button_clicked = col2.button(f"Apply",detail['tender_id'])
            if button_clicked:
               st.success(f"Application submitted successfully.")
        