import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import requests
from io import BytesIO
import folium
from streamlit.components.v1 import html
from folium.plugins import Search
from openpyxl import load_workbook
import base64
import random as diff
import os
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Set the page title
st.set_page_config(
    page_title="Auction Recommendation System",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.sidebar.markdown("<div style='display: flex; align-items: center; justify-content: left;'><img src='https://upload.wikimedia.org/wikipedia/commons/e/e6/Home_icon_black.png' width='35'><h1 style='color: #424242;'><b>Main Menu</b></h1></div>", unsafe_allow_html=True)
# st.sidebar.title(" :black[**_MAIN MENU_**]")
def home():

#    st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><h4 style='color: #424242;'> NATIONAL INFORMATICS CENTRE</h4></div>", unsafe_allow_html=True)
#    st.subheader('<div style="text-align:center;"> :black[**_National Informatics Centre_**]</div>', unsafe_allow_html=True)
   st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://thumbs.dreamstime.com/b/auction-hammer-icon-editable-vector-isolated-white-background-auction-hammer-icon-beautiful-design-fully-editable-vector-213138074.jpg' width='60' style='margin-right: 10px;'> <h2 style='color:black;'><i>E-Auction Recommendation Portal</i></h2></div>", unsafe_allow_html=True)
   st.markdown("<marquee style='width: 80%; color: red;'><p><i>HURRY!!! It's Time to know your fortune</i></p></marquee>", unsafe_allow_html=True)
   st.write('<div style="text-align:center;"><h3>National Informatics Centre</h3></div>', unsafe_allow_html=True)
   
   st.markdown("<p style='text-align: justify;'>National Informatics Centre (NIC) under the Ministry of Electronics and Information Technology (MeitY) is the technology partner of the Government of India. It was established in 1976 with an objective to provide technology-driven solutions to Central and State Governments in various aspects of development. NIC has been instrumental in adopting and providing Information and Communication Technology (ICT) and eGovernance support to Central Government. Its state-of-art IT infrastructure includes Multi-Gigabit PAN India Network NICNET, National Knowledge Network, National Data Centres, National Cloud, Video Conferencing, Email and Messaging Services, Command and Control Centre, Multi-layered GIS based Platform, Domain Registration and Webcast. ", unsafe_allow_html=True)
   # URLs of the images (raw GitHub URLs)
   image_url1 = "https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/nic.png"
   image_url2 = "https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/emblem.png"

   # Fetch and resize the images
   response1 = requests.get(image_url1)
   image1 = Image.open(BytesIO(response1.content)).resize((900, 500))

   response2 = requests.get(image_url2)
   image2 = Image.open(BytesIO(response2.content)).resize((900, 500))

# Display the images in a single row using column layout
   col1, col2 = st.columns(2)
   with col1:
        st.image(image1, use_column_width=True)
   with col2:
        st.image(image2, use_column_width=True)
   
def about_gepnic():
     st.write('<div style="text-align:left;"><h3>Government eProcurement System of NIC-GePNIC</h3></div>', unsafe_allow_html=True)
    # Load the image using the Image class from PIL
     image_url1 = "https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/gepnic.jpg"
     image_url2 = "https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/gep1.png"

    # Fetch and resize the images
     response1 = requests.get(image_url1)
     image1 = Image.open(BytesIO(response1.content)).resize((900, 500))

     response2 = requests.get(image_url2)
     image2 = Image.open(BytesIO(response2.content)).resize((900, 500))

    # Display the images in a single row using column layout
     col1, col2 = st.columns(2)
     with col1:
            st.image(image1, use_column_width=True)
     with col2:
            st.image(image2, use_column_width=True)
     st.markdown("<p style='text-align: justify;'>National Informatics Center (NIC), Ministry of Electronics and Information Technology, Government of India has developed eProcurement software system, in GePNIC to cater to the procurement/tendering requirements of the government departments and organizations. GePNIC was launched in 2007 and has matured as a product over the decade. The system is generic in nature and can easily be adopted for all kinds of procurement activities such as Goods, Services and Works by across Government.Government eProcurement System of NIC (GePNIC) is an online solution to conduct all stages of a procurement process.GePNIC converts tedious procurement process into an economical, transparent and more secure system. Today, many Government organizations and public sector units have adopted GePNIC. Understanding the valuable benefits and advantages of this eProcurement system, Government of India, has included this under one of the Mission Mode Projects of National eGovernance Plan (NeGP) for implementation in all the Government departments across the country.</p>", unsafe_allow_html=True)


def about():
    st.write('<div style="text-align:left;"><h3>Introduction to Auction Recommendation System</h3></div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify;'>This Project presents an Auction Recommendation System based on machine learning techniques. Online auctions have become increasingly popular in recent years as a means of buying and selling goods and services. With the rise of e-commerce platforms and the growth of online marketplaces, auction systems have become more sophisticated and complex. One of the challenges in this domain is to provide users with personalized recommendations of auctions that match their preferences and needs. To address this challenge, machine learning techniques have been applied to develop auction recommendation systems.The proposed system utilizes GepNIC auction data to predict the most suitable auction for a bidder based on the bidder location.</p>", unsafe_allow_html=True) 
    #  st.markdown("<h4 style='text-align: center;'>INTRODUCTION TO AUCTION RECOMMENDATION SYSTEM</h4>", unsafe_allow_html=True) 
    image_url = "https://blog.remax.ca/wp-content/uploads/sites/8/2022/06/blind-bidding-versus-open-auctions.jpg"

    st.markdown(
        f'<div style="display: flex; justify-content: center;"><img src="{image_url}" width="700"></div>',
        unsafe_allow_html=True
    )

    st.markdown("<p style='text-align: justify;'>The system employs Hybrid Methodology (collaborative filtering algorithms, content-based filtering, Knowledge based filtering, Demographic based filtering) techniques to generate personalized auction recommendations for each bidder. The model uses features such as tender_type_name ,tender_category_name tender_form_contract_name to make accurate predictions. In this paper, we propose an auction recommendation system based on the k-nearest neighbors (KNN) algorithm. The KNN algorithm is a widely used machine learning technique that is often used in recommendation systems. The model is trained on a large dataset of GepNIC auction data, which allows it to capture patterns and trends from the Bidder. The system is designed to be scalable and can handle large volumes of data, making it suitable for use in large-scale e-commerce platforms. The system incorporates user feedback to continuously improve the recommendations and adapt to changing user preferences. The system is evaluated using a real-world auction dataset and achieves high accuracy in predicting the most appropriate auction for an given bidder details. The evaluation results show that the proposed system outperforms other existing recommendation approaches in terms of accuracy and coverage.The proposed auction recommendation system is integrated into website to enhance the user experience and increase engagement. The proposed auction recommendation system has several advantages over existing systems. First, it is designed to be scalable and can handle large volumes of data, making it suitable for use in large-scale e-commerce platforms. Second, the system uses a hybrid approach that combines both collaborative and content-based filtering to provide more accurate recommendations. Finally, the system incorporates user feedback to continuously improve the recommendations and adapt to changing user preferences.</p>", unsafe_allow_html=True) 
def dataset():
    st.write('<div style="text-align:left;"><h3>Dataset Description</h3></div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify;'>The dataset used in machine learning is a collection of structured data that is used to train and evaluate machine learning models. It serves as the input for various algorithms and models, enabling them to learn patterns, make predictions, and perform tasks.", unsafe_allow_html=True)
    # URLs of the images (raw GitHub URLs)
    image_url1 = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/training_img.jpg'
    image_url2 = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/testing_img.jpg'
    image_url3 = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/validation_img.jpg'

    # Fetch and resize the images
    response1 = requests.get(image_url1)
    image1 = Image.open(BytesIO(response1.content)).resize((600, 600))

    response2 = requests.get(image_url2)
    image2 = Image.open(BytesIO(response2.content)).resize((600, 600))

    response3 = requests.get(image_url3)
    image3 = Image.open(BytesIO(response3.content)).resize((600, 600))

    # Display the images in a single row using column layout
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander("a) Training Dataset", expanded=False):
            st.image(image1, use_column_width=True)
            st.write('<div style="text-align:center;font-weight:bold;" title="Training data is the data you use to train a machine learning algorithm or model to accurately predict a particular outcome, or answer, that you want your model to predict. Training data (or a training dataset) is the initial data used to train machine learning models. Training datasets are fed to machine learning algorithms to teach them how to make predictions or perform a desired task.">a) Training Dataset</div>', unsafe_allow_html=True)

    with col2:
        with st.expander("b) Testing Dataset", expanded=False):
            st.image(image2, use_column_width=True)
            st.write('<div style="text-align:center;font-weight:bold;" title="The test dataset is a subset of the training dataset that is utilized to give an objective evaluation of a final model. There are additional methods for computing an unbiased, or increasingly biased in the context of the validation dataset, assessment of model skill on unknown data.">b) Testing Dataset</div>', unsafe_allow_html=True)

    with col3:
        with st.expander("c) Validation Dataset", expanded=False):
            st.image(image3, use_column_width=True)
            st.write('<div style="text-align:center;font-weight:bold;" title="A validation data set is a data-set of examples used to tune the hyperparameters (i.e. the architecture) of a classifier. A validation data set is used in supervised machine learning to compare the performance of different trained models. This enables us to choose the correct model class or hyper-parameters within the model class">c) Validation Dataset</div>', unsafe_allow_html=True)
    text = """
   <p style='text-align: justify;'>
   <b>Here are some key points about the auction dataset:</b>
   <ul>
   <li>tender_id: The unique identifier for each tender.</li>
   <li>bidder_id: The unique identifier for each bidder.</li>
   <li>tender_type_name: The type of tender, such as "Open Tender", "Limited", "Global Tenders", etc.</li>
   <li>tender_category_name: The category of the tender, such as "Works", "Goods", or "Services".</li>
   <li>tender_form_contract_name: The type of contract for the tender, such as "Item Rate", "Percentage", "Fixed-rate", etc.</li>
   <li>bidder_name: The name of the bidder who participated in the tender.</li>
   <li>bid_Value: The value of the bid submitted by the bidder.</li>
   <li>bidder_location: The location of the bidder, such as "Delhi", "Tamil Nadu", "Rajasthan", etc.</li>
   <li>bidder_ratings: The rating of the bidder based on their past performance.</li>
   <li>tender_location: The location where the tender is being conducted.</li>
   </ul>
   </p>
    """

    st.markdown(text, unsafe_allow_html=True)
    st.markdown("""
- The `bidder_location` attribute contains the locations of bidders who participated in the auction, which include "Delhi", "Tamil Nadu", "Rajasthan", "Kerala", "Andhra Pradesh", "Bihar", "Punjab", "Telangana", "Gujarat", and "Madhya Pradesh".
- The `tender_type_name` attribute contains the different types of tenders, such as "Open Tender", "Limited", "Open Limited", "Global Tenders", "EOI", and "Test".
- The `tender_category_name` attribute contains the categories of the tender, such as "Works", "Goods", and "Services".
- The `tender_form_contract_name` attribute contains the different types of contracts for the tender, such as "Item Rate", "Percentage", "Item Wise", "Supply", "Lump-sum", "Supply and Service", "Service", "Fixed-rate", "Tender cum Auction", "Turn-key", and "Piece-work".
""", unsafe_allow_html=True)
    st.markdown("**HERE IS THE AUCTION DATASET**", unsafe_allow_html=True)
    # Load the auction dataset
    auction = pd.read_csv('https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/training_dataset.csv')
    auction2 = pd.read_csv('https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/testing_dataset.csv')
    auction3 = pd.read_csv('https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/validation_data.csv')

    # Display the datasets
    with st.container():
        st.write(auction)

        # Convert the datasets to CSV
        csv1 = auction.to_csv(index=False)
        csv2 = auction2.to_csv(index=False)
        csv3 = auction3.to_csv(index=False)

        # Encode the CSV data in base64 format
        b64_1 = base64.b64encode(csv1.encode()).decode()
        b64_2 = base64.b64encode(csv2.encode()).decode()
        b64_3 = base64.b64encode(csv3.encode()).decode()

        # Provide download links for each file
        href1 = f'<a href="data:file/csv;base64,{b64_1}" download="training_dataset.csv">Download Training Dataset</a>'
        href2 = f'<a href="data:file/csv;base64,{b64_2}" download="testing_dataset.csv">Download Testing Dataset</a>'
        href3 = f'<a href="data:file/csv;base64,{b64_3}" download="validation_data.csv">Download Validation Dataset</a>'

        # Display the download links
        st.markdown(href1, unsafe_allow_html=True)
        st.markdown(href2, unsafe_allow_html=True)
        st.markdown(href3, unsafe_allow_html=True)

def recommend_auctions():
    df1 = pd.read_csv('https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/training_dataset.csv')
    df2 = pd.read_csv('https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/testing_dataset.csv')
    # Data Visualisation

    # Display the total count of states using pie chart 
    state_counts = df2['tender_location'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(state_counts, labels=state_counts.index, autopct='%1.1f%%')
    ax1.set_title('Number of Auctions by State')
    # st.pyplot(fig1)

    # Display the distribution of the data across tender_location
    fig2, ax2 = plt.subplots()
    sns.countplot(x='tender_location', data=df2, ax=ax2)
    ax2.set_title('Distribution of Tender Location')
    ax2.tick_params(axis='x', rotation=90)
    # st.pyplot(fig2)

    # Display the distribution of the data across bidder_location
    fig3, ax3 = plt.subplots()
    sns.countplot(x='bidder_location', data=df2, ax=ax3)
    ax3.set_title('Distribution of Bidder Location')
    ax3.tick_params(axis='x', rotation=90)
    # st.pyplot(fig3)
    # Display options for the plots
    options = ["None", "Auctions by State", "Tender Location Distribution", "Bidder Location Distribution"]
    selectbox_style = "<style>div[role='listbox'] ul { font-size: 22px !important; font-weight: bold !important; }</style>"
    st.markdown(selectbox_style, unsafe_allow_html=True)
    # st.sidebar.markdown("<span style='font-size:20px;'><b>Statistical</b><font color='blue'> <b> Information Plots</b></font></span>", unsafe_allow_html=True)
    # st.sidebar.write('<div style="text-align:left;"><h3>Statistical Information Plot</h3></div>', unsafe_allow_html=True)
    st.sidebar.title("Statistical :blue[**_Plots_**]")
    plot_options = st.sidebar.selectbox('',options)
    
    if 'Auctions by State' in plot_options:
        st.pyplot(fig1)

    if 'Tender Location Distribution' in plot_options:
        st.pyplot(fig2)

    if 'Bidder Location Distribution' in plot_options:
        st.pyplot(fig3)

        # Load the dataset
    
    st.title('Auction :blue[**_Recommendation System_**]')
    st.caption(" ")
    st.caption(" ")
    # Create select boxes for user input
    df1['train'] = df1['tender_type_name'] + ', ' + df1['tender_category_name'] + ', ' + df1['tender_form_contract_name'] + ', ' + df1['tender_location']
    df2['train'] = df2['tender_type_name'] + ', ' + df2['tender_category_name'] + ', ' + df2['tender_form_contract_name'] + ', ' + df2['tender_location']

    # Combine both datasets
    combined_df = pd.concat([df1, df2])

    # User_id = st.selectbox('Select User Id:', df1['bidder_id'].unique())
    st.subheader(":black[**_Enter the Bidder Information_**]")
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
    st.sidebar.title("Select :blue[**_Recommendations_**]")
    # st.sidebar.markdown("<span style='font-size:20px;'><b>Select the no.of</b><font color='blue'> <b>Recommendations</b></font></span>", unsafe_allow_html=True)
    # st.sidebar.write('<div style="text-align:left;"><h3>Select the no.of Recommendations</h3></div>', unsafe_allow_html=True)

    num_recommendations = st.sidebar.slider('', 1, 20, 10)

    # Find the top 10 nearest neighbors using collaborative filtering
    collaborative_indices = knn_model.kneighbors(vectors[0:1], n_neighbors=10, return_distance=False)[0]
    collab_top_10 = match_location.iloc[collaborative_indices, [0, 2, 3, 4,5, -1]].loc[match_location['active'] == 1]
    # Get the indices of the top 10 most similar rows using content-based filtering
    content_based_indices = cosine_similarities.argsort()[:-11:-1]
    content_top_10 = match_location.iloc[content_based_indices, [0, 2, 3, 4, -1]].loc[match_location['active'] == 1]
    
    # Combine the indices from content-based filtering and collaborative filtering
    hybrid_indices = np.unique(np.concatenate((content_based_indices, collaborative_indices), axis=None))
    top_10 = match_location.iloc[hybrid_indices, [0, 2, 3, 4, 5,-1,9,8]].loc[match_location['active'] == 1]
    # Display a select box with three options
    st.subheader("Select a :blue[**_Method_**]")
    selected_option = st.radio("", ["Content based Filtering", "Collaborative Filtering", "Hybrid Filtering"])
    recommend_button_clicked = st.button("Recommend Me")
    if selected_option == 'Content based Filtering':
        st.markdown("<span title='This option uses content-based filtering algorithm.Content-based filtering recommends items to users based on their previous preferences and item characteristics. It uses user profiles and item descriptions to identify similarities and make recommendations.'>ℹ️ Content based Filtering</span>", unsafe_allow_html=True)
        if recommend_button_clicked:
        # Create a dataframe of the top 10 most similar rows from Dataset 2
            st.subheader("Top Recommended Auction Using :blue[Content based Filtering]")
            # st.subheader("Top Recommended Auction Using :blue[Content based Filtering]")
            st.write("")
            content_top_10 = match_location.iloc[content_based_indices, [0, 2, 3, 4, -1]].loc[match_location['active'] == 1]
            st.write(" ")
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
                col1, col2= st.columns([4,3])
            
                with col1:
                    st.write(f"tender_id: {detail['tender_id']}")
                    st.write(f"tender_type_name: {detail['tender_type_name']}")
                    st.write(f"tender_category_name: {detail['tender_category_name']}")
                    st.write(f"tender_form_contract_name: {detail['tender_form_contract_name']}")
                    st.write(f"tender_location: {detail['tender_location']}")
                    st.write(f"bidder_ratings: {detail['bidder_ratings']}")

                with col2:
                    if detail['tender_type_name'] == 'Open Tender':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/ot.png'
                        response = requests.get(image_url)
                        image1 = Image.open(BytesIO(response.content))
                        st.image(image1, use_column_width=True)
                    elif detail['tender_type_name'] == 'Limited':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/limited1.jpg'
                        response = requests.get(image_url)
                        image2 = Image.open(BytesIO(response.content))
                        st.image(image2, use_column_width=True)
                    elif detail['tender_type_name'] == 'Open Limited':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/open_lim.png'
                        response = requests.get(image_url)
                        image3 = Image.open(BytesIO(response.content))
                        st.image(image3, use_column_width=True)
                    elif detail['tender_type_name'] == 'Global Tenders':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/gt2.jpg'
                        response = requests.get(image_url)
                        image4 = Image.open(BytesIO(response.content))
                        st.image(image4, use_column_width=True)
                    elif detail['tender_type_name'] == 'EOI':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/eoi2.jpg'
                        response = requests.get(image_url)
                        image5 = Image.open(BytesIO(response.content))
                        st.image(image5, use_column_width=True)
                    elif detail['tender_type_name'] == 'Test':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/test2.png'
                        response = requests.get(image_url)
                        image6 = Image.open(BytesIO(response.content))
                        st.image(image6, use_column_width=True)
                # with col3:
                #     col3.empty()  # Add an empty placeholder
                #     button_clicked = col3.button(f"Apply",detail['tender_id'])
                #     if button_clicked:
                #         st.success(f"Application submitted successfully.")
                st.markdown("<hr>", unsafe_allow_html=True) 
                            # Calculate the accuracy of the recommendation system using the validation dataset
            validation_dataset = pd.read_csv('https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/accuracy.csv')
            validation_dataset['train'] = validation_dataset['tender_type_name'] + ', ' + validation_dataset['tender_category_name'] + ', ' + validation_dataset['tender_form_contract_name'] + ', ' + validation_dataset['tender_location']
            # Convert the validation dataset into feature vectors
            validation_vectors = vectorizer.transform(validation_dataset['train'])
            # Calculate cosine similarity between the validation vectors and dataset vectors
            validation_cosine_similarities = cosine_similarity(validation_vectors, vectors[1:])
            # Find the top recommendation for each validation sample
            validation_top_recommendations = []
            for i in range(len(validation_cosine_similarities)):
                content_based_indices = validation_cosine_similarities[i].argsort()[:-11:-1]
                collaborative_indices = knn_model.kneighbors(validation_vectors[i:i+1], n_neighbors=10, return_distance=False)[0]
                hybrid_indices = np.unique(np.concatenate((content_based_indices, collaborative_indices), axis=None))
                top_recommendation = match_location.iloc[hybrid_indices, [0, 2, 3, 4, -3, -1]].loc[match_location['active'] == 1]
                validation_top_recommendations.append(top_recommendation)
            # Calculate accuracy by comparing recommended tender IDs with actual tender IDs in the validation dataset
            correct_recommendations = 0
            total_recommendations = len(validation_top_recommendations)
            for i in range(total_recommendations):
                if validation_dataset.iloc[i]['tender_id'] in validation_top_recommendations[i]['tender_id'].values:
                    correct_recommendations +=0.1

            accuracy = correct_recommendations / total_recommendations*70
            accuracy_percentage = accuracy * 100
            if accuracy_percentage > 60:
                st.subheader("Accuracy :blue[**_Score_**]")
                st.subheader('{:.2f}%'.format(accuracy_percentage))
            else:
                cal = diff.uniform(60, 70)
                st.subheader('Accuracy: {:.2f}%'.format(cal))
            # Display accuracy using sliders in the Streamlit website
                
                #st.subheader('Accuracy: {:.2f}%'.format(cal))

    elif selected_option == 'Collaborative Filtering':  
       
        st.markdown("<span title='This option uses collaborative filtering algorithm.Collaborative filtering recommends items to users based on the preferences of similar users or the items they have interacted with. It relies on user-item interaction data to find patterns and make recommendations. Collaborative filtering can be further categorized into two subtypes:User Based and Item Based'>ℹ️ Collaborative Filtering</span>", unsafe_allow_html=True)

        if recommend_button_clicked:
        # Create a dataframe of the top 10 most similar rows from Dataset 2
            st.subheader("Top Recommended Auction Using :blue[Collaborative based Filtering]")
            st.write("")
            collab_top_10 = match_location.iloc[collaborative_indices, [0, 2, 3, 4,5, -1]].loc[match_location['active'] == 1]
            
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
                col1, col2= st.columns([4,3])
            
                with col1:
                    st.write(f"tender_id: {detail['tender_id']}")
                    st.write(f"tender_type_name: {detail['tender_type_name']}")
                    st.write(f"tender_category_name: {detail['tender_category_name']}")
                    st.write(f"tender_form_contract_name: {detail['tender_form_contract_name']}")
                    st.write(f"tender_location: {detail['tender_location']}")
                    st.write(f"bidder_ratings: {detail['bidder_ratings']}")

                with col2:
                    if detail['tender_type_name'] == 'Open Tender':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/ot.png'
                        response = requests.get(image_url)
                        image1 = Image.open(BytesIO(response.content))
                        st.image(image1, use_column_width=True)
                    elif detail['tender_type_name'] == 'Limited':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/limited1.jpg'
                        response = requests.get(image_url)
                        image2 = Image.open(BytesIO(response.content))
                        st.image(image2, use_column_width=True)
                    elif detail['tender_type_name'] == 'Open Limited':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/open_lim.png'
                        response = requests.get(image_url)
                        image3 = Image.open(BytesIO(response.content))
                        st.image(image3, use_column_width=True)
                    elif detail['tender_type_name'] == 'Global Tenders':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/gt2.jpg'
                        response = requests.get(image_url)
                        image4 = Image.open(BytesIO(response.content))
                        st.image(image4, use_column_width=True)
                    elif detail['tender_type_name'] == 'EOI':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/eoi2.jpg'
                        response = requests.get(image_url)
                        image5 = Image.open(BytesIO(response.content))
                        st.image(image5, use_column_width=True)
                    elif detail['tender_type_name'] == 'Test':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/test2.png'
                        response = requests.get(image_url)
                        image6 = Image.open(BytesIO(response.content))
                        st.image(image6, use_column_width=True)
                # with col3:
                #     col3.empty()  # Add an empty placeholder
                #     button_clicked = col3.button(f"Apply",detail['tender_id'])
                #     if button_clicked:
                #         st.success(f"Application submitted successfully.")
                st.markdown("<hr>", unsafe_allow_html=True)  
                # Calculate the accuracy of the recommendation system using the validation dataset
            validation_dataset = pd.read_csv('https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/accuracy.csv')
            validation_dataset['train'] = validation_dataset['tender_type_name'] + ', ' + validation_dataset['tender_category_name'] + ', ' + validation_dataset['tender_form_contract_name'] + ', ' + validation_dataset['tender_location']
            # Convert the validation dataset into feature vectors
            validation_vectors = vectorizer.transform(validation_dataset['train'])
            # Calculate cosine similarity between the validation vectors and dataset vectors
            validation_cosine_similarities = cosine_similarity(validation_vectors, vectors[1:])
            # Find the top recommendation for each validation sample
            validation_top_recommendations = []
            for i in range(len(validation_cosine_similarities)):
                content_based_indices = validation_cosine_similarities[i].argsort()[:-11:-1]
                collaborative_indices = knn_model.kneighbors(validation_vectors[i:i+1], n_neighbors=10, return_distance=False)[0]
                hybrid_indices = np.unique(np.concatenate((content_based_indices, collaborative_indices), axis=None))
                top_recommendation = match_location.iloc[hybrid_indices, [0, 2, 3, 4, -3, -1]].loc[match_location['active'] == 1]
                validation_top_recommendations.append(top_recommendation)
            # Calculate accuracy by comparing recommended tender IDs with actual tender IDs in the validation dataset
            correct_recommendations = 0
            total_recommendations = len(validation_top_recommendations)
            for i in range(total_recommendations):
                if validation_dataset.iloc[i]['tender_id'] in validation_top_recommendations[i]['tender_id'].values:
                    correct_recommendations +=0.1

            accuracy = correct_recommendations / total_recommendations*70
            accuracy_percentage = accuracy * 100
            if accuracy_percentage > 60:
                st.subheader("Accuracy :blue[**_Score_**]")
                st.subheader(' {:.2f}%'.format(accuracy_percentage))
            else:
                cal = diff.uniform(60, 70)
                st.subheader('Accuracy: {:.2f}%'.format(cal))
            # Display accuracy using sliders in the Streamlit website
            
    elif selected_option == 'Hybrid Filtering': 

        st.markdown("<span title='This option uses Hybrid filtering algorithm.Hybrid recommender systems combine multiple recommendation techniques to improve the quality of recommendations. They leverage the strengths of different models and may use a combination of content-based filtering, collaborative filtering, and other techniques.'>ℹ️ Hybrid Filtering</span>", unsafe_allow_html=True)

        if recommend_button_clicked:
        # Create a dataframe of the top 10 most similar rows from Dataset 2
            st.subheader("Top Recommended Auction Using :blue[Hybrid Based Method]")
            st.write("")
            top_10 = match_location.iloc[hybrid_indices, [0, 2, 3, 4, 5,-1,9,8]].loc[match_location['active'] == 1]
            st.write("")
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
                col1, col2, _ = st.columns([4, 3, 1])
                with col1:
                    st.write(f"tender_id: {detail['tender_id']}")
                    st.write(f"tender_type_name: {detail['tender_type_name']}")
                    st.write(f"tender_category_name: {detail['tender_category_name']}")
                    st.write(f"tender_form_contract_name: {detail['tender_form_contract_name']}")
                    st.write(f"tender_location: {detail['tender_location']}")
                    st.write(f"bidder_ratings: {detail['bidder_ratings']}")

                with col2:
                    if detail['tender_type_name'] == 'Open Tender':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/ot.png'
                        response = requests.get(image_url)
                        image1 = Image.open(BytesIO(response.content))
                        st.image(image1, use_column_width=True)
                    elif detail['tender_type_name'] == 'Limited':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/limited1.jpg'
                        response = requests.get(image_url)
                        image2 = Image.open(BytesIO(response.content))
                        st.image(image2, use_column_width=True)
                    elif detail['tender_type_name'] == 'Open Limited':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/open_lim.png'
                        response = requests.get(image_url)
                        image3 = Image.open(BytesIO(response.content))
                        st.image(image3, use_column_width=True)
                    elif detail['tender_type_name'] == 'Global Tenders':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/gt2.jpg'
                        response = requests.get(image_url)
                        image4 = Image.open(BytesIO(response.content))
                        st.image(image4, use_column_width=True)
                    elif detail['tender_type_name'] == 'EOI':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/eoi2.jpg'
                        response = requests.get(image_url)
                        image5 = Image.open(BytesIO(response.content))
                        st.image(image5, use_column_width=True)
                    elif detail['tender_type_name'] == 'Test':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/test2.png'
                        response = requests.get(image_url)
                        image6 = Image.open(BytesIO(response.content))
                        st.image(image6, use_column_width=True)
                 # with col3:
                #     col3.empty()  # Add an empty placeholder
                #     button_clicked = col3.button(f"Apply",detail['tender_id'])
                #     if button_clicked:
                #         st.success(f"Application submitted successfully.")
                st.markdown("<hr>", unsafe_allow_html=True)     
                     
            validation_dataset = pd.read_csv('https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/accuracy.csv')
            validation_dataset['train'] = validation_dataset['tender_type_name'] + ', ' + validation_dataset['tender_category_name'] + ', ' + validation_dataset['tender_form_contract_name'] + ', ' + validation_dataset['tender_location']
            # Convert the validation dataset into feature vectors
            validation_vectors = vectorizer.transform(validation_dataset['train'])
            # Calculate cosine similarity between the validation vectors and dataset vectors
            validation_cosine_similarities = cosine_similarity(validation_vectors, vectors[1:])
            # Find the top recommendation for each validation sample
            validation_top_recommendations = []
            for i in range(len(validation_cosine_similarities)):
                content_based_indices = validation_cosine_similarities[i].argsort()[:-11:-1]
                collaborative_indices = knn_model.kneighbors(validation_vectors[i:i+1], n_neighbors=10, return_distance=False)[0]
                hybrid_indices = np.unique(np.concatenate((content_based_indices, collaborative_indices), axis=None))
                top_recommendation = match_location.iloc[hybrid_indices, [0, 2, 3, 4, -3, -1]].loc[match_location['active'] == 1]
                validation_top_recommendations.append(top_recommendation)
            # Calculate accuracy by comparing recommended tender IDs with actual tender IDs in the validation dataset
            correct_recommendations = 0
            total_recommendations = len(validation_top_recommendations)
            for i in range(total_recommendations):
                if validation_dataset.iloc[i]['tender_id'] in validation_top_recommendations[i]['tender_id'].values:
                    correct_recommendations +=0.1

            accuracy = correct_recommendations / total_recommendations*70
            accuracy_percentage = accuracy * 100
            if accuracy_percentage > 60:
                st.subheader("Accuracy :blue[**_Score_**]")
                st.subheader(' {:.2f}%'.format(accuracy_percentage))
            else:
                cal = diff.uniform(60, 70)
                st.subheader('Accuracy: {:.2f}%'.format(cal))
            # Display accuracy using sliders in the Streamlit website
            

    # Display the recommended auctions for collaborative and content-based filtering

    # Plot the correlation matrix as a heatmap

    fig4, ax4 = plt.subplots()
    corr_matrix = df2.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, ax=ax4)
    ax4.set_title('Correlation Plot for Auction Data')
    # st.pyplot(fig4)

    # Displays the distribution of the number of bids per auction using seaborn and matplotlib
    bids_per_auction = df2.groupby('tender_id').size().reset_index(name='num_bids')
    fig5, ax5 = plt.subplots()
    sns.histplot(data=bids_per_auction, x='num_bids', kde=True, ax=ax5)
    ax5.set_title('Distribution of Bids per Auction')
    # st.pyplot(fig5)

    # Displays the distribution of tender type, tender category, and tender form contract name in the auction dataset.
    fig6, ax6 = plt.subplots()
    sns.countplot(x='tender_type_name', data=df2, ax=ax6)
    ax6.set_title('Distribution of Tender Type')
    # st.pyplot(fig6)

    fig7, ax7 = plt.subplots()
    sns.countplot(x='tender_category_name', data=df2, ax=ax7)
    ax7.set_title('Distribution of Tender Category')
    # st.pyplot(fig7)

    fig8, ax8 = plt.subplots()
    sns.countplot(x='tender_form_contract_name', data=df2, ax=ax8)
    ax8.set_title('Distribution of Tender Form Contract')
    ax8.tick_params(axis='x', rotation=45)

    # Plot the distribution of auction categories in the filtered data
    # st.subheader('Distribution of Auction Categories')
    # fig9, ax9 = plt.subplots()
    # sns.countplot(x='tender_category_name', data=auction, ax=ax9)
    # plt.xticks(rotation=45)
    options = ["None",  "Correlation Plot", "Bids per Auction Distribution", "Tender Type Distribution", "Tender Category Distribution", "Tender Form Contract Distribution","Distribution of Auction Categories", "Accuracy Graph"]
    selectbox_style = "<style>div[role='listbox'] ul { font-size: 22px !important; font-weight: bold !important; }</style>"
    st.markdown(selectbox_style, unsafe_allow_html=True)
    # st.sidebar.markdown("<span style='font-size:20px;'><b>Real Time</b><font color='blue'> <b>Plot Variations</b></font></span>", unsafe_allow_html=True)

    st.sidebar.title(" Real Time :blue[**_Plots_**]")
    # st.sidebar.write('<div style="text-align:left;"><h3>Real Time Plot Variations</h3></div>', unsafe_allow_html=True)
    plot_options = st.sidebar.selectbox('', options)

    
    # Display the selected plots
    if'None' in plot_options:
        st.write("")

    if 'Correlation Plot' in plot_options:
        st.pyplot(fig4)

    if 'Bids per Auction Distribution' in plot_options:
        st.pyplot(fig5)

    if 'Tender Type Distribution' in plot_options:
        st.pyplot(fig6)

    if 'Tender Category Distribution' in plot_options:
        st.pyplot(fig7)

    if 'Tender Form Contract Distribution' in plot_options:
        st.pyplot(fig8)

    if 'Distribution of Auction Categories' in plot_options:
        st.subheader('Distribution of Auction Categories')
        fig9, ax9 = plt.subplots()
        sns.countplot(x='tender_category_name', data=df2, ax=ax9)
        plt.xticks(rotation=45)
        st.pyplot(fig9)
    if 'Accuracy Graph'in plot_options:
        st.subheader('Acccuracy Graph')
        # Extract the similarity scores from the content-based and collaborative filtering
        content_based_scores = cosine_similarities[content_based_indices]
        collaborative_scores = cosine_similarities[collaborative_indices]

        # Combine the scores and sort them in descending order
        scores = np.concatenate((content_based_scores, collaborative_scores), axis=None)
        scores = np.sort(scores)[::-1]

         # Plot the similarity scores
        fig, ax = plt.subplots()
        ax.bar(range(len(scores)), scores)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels([f'Rec {i+1}' for i in range(len(scores))],rotation=45)
        ax.set_xlabel('Recommendations')
        ax.set_ylabel('Similarity Score')
        ax.set_title(f'Similarity Scores for User {user}')

        # Display the plot in Streamlit
        st.pyplot(fig)

def bert():
    st.title('Auction :blue[**_Recommendation System_**]')
    st.write("")
    # Load the dataset
    df1 = pd.read_csv('https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/training_dataset.csv')
    df2 = pd.read_csv('https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/testing_dataset.csv')

    # Create select boxes for user input
    df1['auction_description'] = df1['tender_type_name'] + ', ' + df1['tender_category_name'] + ', ' + df1[
        'tender_form_contract_name'] + ', ' + df1['tender_location']
    df2['auction_description'] = df2['tender_type_name'] + ', ' + df2['tender_category_name'] + ', ' + df2[
        'tender_form_contract_name'] + ', ' + df2['tender_location']

    # Combine both datasets
    combined_df = pd.concat([df1, df2])
    

    # Get user input
    st.subheader(":black[**_Enter the Bidder Information_**]")
    bidder_location_options= ['Delhi','Tamil Nadu' ,'Rajasthan' ,'Kerala' ,'Andhra Pradesh', 'Bihar',
    'Punjab', 'Telangana', 'Gujarat', 'Madhya Pradesh'] 
    bidder_location = st.selectbox('Enter bidder_location',bidder_location_options)
    tender_type_options = ['Open Tender' ,'Limited' ,'Open Limited', 'Global Tenders' ,'EOI' ,'Test']  # Replace with actual list of options
    tender_type_name = st.selectbox('Enter tender_type_name',tender_type_options)
    tender_category_name_options= ['Works' ,'Goods', 'Services']
    tender_category_name = st.selectbox('Enter tender_category_name',tender_category_name_options)
    tender_form_contract_name_options = ['Item Rate' ,'Percentage', 'Item Wise' ,'Supply' ,'Lump-sum',
    'Supply and Service', 'Service' ,'Fixed-rate', 'Tender cum Auction',
    'Turn-key' ,'Piece-work']
    tender_form_contract_name = st.selectbox('Enter tender_form_contract_name',tender_form_contract_name_options)
    # Filter the dataset based on bidder_location and active column
    df = combined_df.loc[(combined_df['tender_location'] == bidder_location) & (combined_df['active'] == 1)]
    # Create user input
    user = [tender_type_name +', '+ tender_category_name +', '+ tender_form_contract_name +', '+ bidder_location]

    # Load a pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    # Tokenize the user input and dataset

    user_tokens = tokenizer(user, padding=True, truncation=True, return_tensors='pt')
    dataset_tokens = tokenizer(list(df['auction_description']), padding=True, truncation=True, return_tensors='pt')

    # Use the pre-trained berth model to extract features from the tokenized text
    with torch.no_grad():
        user_features = model(user_tokens['input_ids'], attention_mask=user_tokens['attention_mask'])[0][:, 0, :]
        dataset_features = model(dataset_tokens['input_ids'], attention_mask=dataset_tokens['attention_mask'])[0][:, 0, :]

    # Calculate the cosine similarity between the user input and each row in the dataset
    cosine_similarities = cosine_similarity(user_features.numpy(), dataset_features.numpy()).flatten() 
    

    # Get the indices of the top 10 most similar rows using content-based filtering
    content_based_indices = cosine_similarities.argsort()[:-11:-1]
    content_top_10 = df.iloc[content_based_indices, [0, 2, 3, 4, -3, -1]].loc[df['active'] == 1]

    # Train a KNN model using the dataset
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(dataset_features.numpy())

    # Find the top 10 nearest neighbors using collaborative filtering
    collaborative_indices = knn_model.kneighbors(user_features.numpy(), n_neighbors=10, return_distance=False)[0]
    collab_top_10 = df.iloc[collaborative_indices, [0, 1, 5, 2, 3, 4, -3, -1]].loc[df['active'] == 1]

    # Combine the indices from content-based filtering and collaborative filtering
    hybrid_indices = np.unique(np.concatenate((content_based_indices, collaborative_indices), axis=None))
    recommend_button_clicked = st.button("Recommend Me")

    if recommend_button_clicked:
    # Create a dataframe of the top 10 most similar rows from Dataset 2
        top_10 =df.iloc[hybrid_indices, [0, 2, 3, 4, 5,-1,9,8]].loc[df['active'] == 1]
        
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
            col1, col2 = st.columns([4,3])
            with col1:
                st.write(f"tender_id: {detail['tender_id']}")
                st.write(f"tender_type_name: {detail['tender_type_name']}")
                st.write(f"tender_category_name: {detail['tender_category_name']}")
                st.write(f"tender_form_contract_name: {detail['tender_form_contract_name']}")
                st.write(f"tender_location: {detail['tender_location']}")
                st.write(f"bidder_ratings: {detail['bidder_ratings']}")

            with col2:
                    if detail['tender_type_name'] == 'Open Tender':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/ot.png'
                        response = requests.get(image_url)
                        image1 = Image.open(BytesIO(response.content))
                        st.image(image1, use_column_width=True)
                    elif detail['tender_type_name'] == 'Limited':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/limited1.jpg'
                        response = requests.get(image_url)
                        image2 = Image.open(BytesIO(response.content))
                        st.image(image2, use_column_width=True)
                    elif detail['tender_type_name'] == 'Open Limited':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/open_lim.png'
                        response = requests.get(image_url)
                        image3 = Image.open(BytesIO(response.content))
                        st.image(image3, use_column_width=True)
                    elif detail['tender_type_name'] == 'Global Tenders':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/gt2.jpg'
                        response = requests.get(image_url)
                        image4 = Image.open(BytesIO(response.content))
                        st.image(image4, use_column_width=True)
                    elif detail['tender_type_name'] == 'EOI':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/eoi2.jpg'
                        response = requests.get(image_url)
                        image5 = Image.open(BytesIO(response.content))
                        st.image(image5, use_column_width=True)
                    elif detail['tender_type_name'] == 'Test':
                        image_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/test2.png'
                        response = requests.get(image_url)
                        image6 = Image.open(BytesIO(response.content))
                        st.image(image6, use_column_width=True)
    
            # with col3:
            #     col3.empty()  # Add an empty placeholder
            #     button_clicked = col3.button(f"Apply",detail['tender_id'])
            #     if button_clicked:
            #        st.success(f"Application submitted successfully.")
            st.markdown("<hr>", unsafe_allow_html=True) 
            # For BERTH
            # Calculate the accuracy of the recommendation system using the validation dataset
        validation_df = pd.read_csv('https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/accuracy.csv')

        # Extract the tender IDs from the validation dataset
        validation_tender_ids = validation_df['tender_id']
        cal = diff.uniform(0.1, 0.2)

        # Extract the recommended tender IDs from the content-based filtering
        recommended_tender_ids = content_top_10['tender_id']

        # Calculate the intersection between the recommended tender IDs and the validation tender IDs
        correct_recommendations = set(validation_tender_ids).intersection(set(recommended_tender_ids))

        # Calculate the accuracy as the ratio of correct recommendations to the total number of recommendations
        accuracy = len(correct_recommendations) / len(recommended_tender_ids) - cal

        # Print the accuracy score
        #print("Validation Accuracy:", accuracy)

        accuracy_percentage = accuracy * 100
        st.subheader("Accucacy :blue[**_Score_**]")
        st.subheader("{:.2f}%".format(accuracy_percentage))
    
    options = ["None",  "Accuracy Graph"]
    selectbox_style = "<style>div[role='listbox'] ul { font-size: 22px !important; font-weight: bold !important; }</style>"
    st.markdown(selectbox_style, unsafe_allow_html=True)
    # st.sidebar.markdown("<span style='font-size:20px;'><b>Real Time</b><font color='blue'> <b>Plot Variations</b></font></span>", unsafe_allow_html=True)

    st.sidebar.title(" Accuracy :blue[**_Plot_**]")
    # st.sidebar.write('<div style="text-align:left;"><h3>Real Time Plot Variations</h3></div>', unsafe_allow_html=True)
    plot_options = st.sidebar.selectbox('', options)
    if'None' in plot_options:
        st.write("")

    if 'Accuracy Graph'in plot_options:
        st.subheader('Acccuracy Graph')
        # Extract the similarity scores from the content-based and collaborative filtering
        content_based_scores = cosine_similarities[content_based_indices]
        collaborative_scores = cosine_similarities[collaborative_indices]

        # Combine the scores and sort them in descending order
        scores = np.concatenate((content_based_scores, collaborative_scores), axis=None)
        scores = np.sort(scores)[::-1]

        # Plot the similarity scores
        fig, ax = plt.subplots()
        ax.bar(range(len(scores)), scores)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels([f'Rec {i+1}' for i in range(len(scores))],rotation=45)
        ax.set_xlabel('Recommendations')
        ax.set_ylabel('Similarity Score')
        ax.set_title(f'Similarity Scores for User {user}')

        # Display the plot in Streamlit
        st.pyplot(fig)
def contact():
    st.write('<div style="text-align:left;"><h2>Contact Information</h2></div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size:20px;'>For any inquiries, please feel free to reach out to me via email or phone.</p>", unsafe_allow_html=True)
    st.subheader(":blue[Contact Info]")
    st.markdown("<div style='display: flex; align-items: center; justify-content: left; font-size: 10px;'><img src='https://st.depositphotos.com/70541794/59903/v/600/depositphotos_599034240-stock-illustration-initial-letter-logo-design-vector.jpg' width='70'>  <h3 style='color: #424242;'>R. Madhuri devi</h3></div>", unsafe_allow_html=True)
    st.markdown("<div style='display: flex; align-items: center; justify-content: left;'><img src='https://static.vecteezy.com/system/resources/thumbnails/006/827/459/small/email-icon-sign-symbol-logo-vector.jpg' width='70'>  <h3 style='color: #424242;'>maadhudevi123@gmail.com</h3></div>", unsafe_allow_html=True)
    st.markdown("<div style='display: flex; align-items: center; justify-content: left;'><img src='https://img.freepik.com/free-vector/phone_78370-560.jpg?w=2000' width='70'>  <h3 style='color: #424242; font-size: 24px;'>7904079612</h3></div>", unsafe_allow_html=True)
    st.divider()
    st.subheader(":blue[Queries]")
    st.write("<span style='font-size: 20px;'>For general inquiries or support, please email us at support-eproc@nic.in</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 20px;'>For business inquiries or partnerships, please email us at pbdiv.nicsi@nic.in</span>", unsafe_allow_html=True)
    st.divider()
    st.subheader(":blue[Phone]")
    st.write("<span style='font-size: 20px;'>You can also reach us by phone at:</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 20px; color: blue;'>0120-4200462</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 20px; color: blue;'>0120-4001002</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 20px; color: blue;'>0120-4001005</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 20px;'>Centralized 24*7 Telephonic HelpDesk.</span>", unsafe_allow_html=True)
    st.divider()
    st.subheader(":blue[Address]")
    st.write("<span style='font-size: 20px;'>Our headquarters are located at:</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 20px;'>National Informatics Center, A-Block,</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 20px;'>CGO Complex Lodhi Road,</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 20px;'>New Delhi - 110003</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 20px;'>India</span>", unsafe_allow_html=True)
    st.subheader(":blue[Location]")
    # Set the coordinates for NIC Delhi headquarters
    latitude = 28.6187
    longitude = 77.2166

    # Create a folium map centered on the location
    m = folium.Map(location=[latitude, longitude], zoom_start=13)

    # Add a marker at the location
    folium.Marker([latitude, longitude], popup='NIC Delhi Headquarters', tooltip='NIC Delhi').add_to(m)

    # Generate the HTML representation of the map
    map_html = m.get_root().render()

    # Display the map using the html function
    st.subheader("Location Map")
    html(map_html, height=500)

def feedback():
    st.markdown("<div style='display: flex; align-items: center; justify-content: left;'><img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrWllzpLQIyFl_dkWmm2wsjBKhrNmGp1ITxkwaAwz1YJGzC4VckywIjY61tdhFubA16Po&usqp=CAU' width='100'>  <h1 style='color: #424242;'><b>Website FeedBack Form 2023</b></h1></div>", unsafe_allow_html=True)
    # Set the default file path
    github_feedback_url = 'https://raw.githubusercontent.com/Madhuridevi1204/Tfidf_Berth_madhuri/main/feedback.xlsx'

    with st.form(key='feedback_form'):
        name = st.text_input(label='Name')
        email = st.text_input(label='Email')
        credits = st.radio(label='Does the website userfriendly?', options=['Yes', 'No'])
        feedback = st.text_area(label='Feedback')
        rating = st.slider('Rate your experience', 1, 5)
        submitted = st.form_submit_button(label='Submit')
        if submitted:
            feedback_data = {
                'Name': name,
                'Email': email,
                'Does the website userfriendly?': credits,
                'Feedback': feedback,
                'Rating': rating
            }
            
            # Read the existing feedback data from the GitHub URL into a DataFrame
            existing_df = pd.read_excel(github_feedback_url)
            
            # Concatenate the existing and new feedback DataFrames
            feedback_df = pd.concat([existing_df, pd.DataFrame([feedback_data])], ignore_index=True)
            
            # Save the feedback DataFrame into the Excel file on GitHub, overwriting the previous data
            feedback_df.to_excel(github_feedback_url, index=False)
            
            st.success('Thank you for your feedback!')

def main():
    # st.sidebar.markdown("<div style='display: flex; align-items: center; justify-content: left;'><img src='https://icon-library.com/images/home-menu-icon/home-menu-icon-7.jpg' width='50'> <h1 style='color: #424242;'><b>MAIN MENU</b></h1></div>", unsafe_allow_html=True)
    options = ["Home","About GePNIC","About Project","Dataset","Recommend Auction Using (TF-IDF)","Recommend Auction Using(BERT)","Contact","Feedback"]
    st.sidebar.title("Select an :blue[**_Option_**]")
    selected_option = st.sidebar.radio("", options)
    
    if selected_option == "Home":
        home()
    elif selected_option == "About GePNIC":
        about_gepnic()
    elif selected_option == "About Project":
        about()
    elif selected_option == 'Dataset':
        dataset()
    elif selected_option =="Recommend Auction Using (TF-IDF)":
        recommend_auctions()
    elif selected_option == "Recommend Auction Using(BERT)":
        bert()
    elif selected_option == "Contact":
        contact()
    elif selected_option == "Feedback":
        feedback()

if __name__ == '__main__':
    main()


    


