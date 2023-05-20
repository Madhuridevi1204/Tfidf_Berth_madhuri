import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
import torch

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
st.sidebar.markdown("<div style='display: flex; align-items: center; justify-content: left;'><img src='https://icon-library.com/images/home-menu-icon/home-menu-icon-7.jpg' width='50'> <h1 style='color: #424242;'><b>MAIN MENU</b></h1></div>", unsafe_allow_html=True)

# st.sidebar.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='' width='50' style='margin-right: 10px;'> <h1 style='color: #424242;'>MAIN MENU</h1></div>", unsafe_allow_html=True)

# options = ["RECOMMEND AUCTION", "HOME", "ABOUT", "DATASET", "CONTACT", "FEEDBACK"]
# selected_option = st.sidebar.radio("", options)
# if selected_option == "RECOMMEND AUCTION":
#    st.write("")
   
# elif selected_option == "HOME":
# #    st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://img.freepik.com/free-icon/legal-hammer-symbol_318-64606.jpg?size=626&ext=jpg&uid=R101051765&ga=GA1.2.1088004906.1682729121&semt=robertav1_2_sidr' width='50' style='margin-right: 10px;'> <h2 style='color: #424242;'>E-ONLINE AUCTION RECOMMENDATION</h2></div>", unsafe_allow_html=True)
# #    st.markdown("<marquee style='width: 80%; color: red;'><p>HURRY!!! It's Time to get the pace</p></marquee>", unsafe_allow_html=True)
#    st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><h1 style='color: #424242;'> GEPNIC E-PROCUREMENT</h1></div>", unsafe_allow_html=True)
#    st.markdown("<h1 style='text-align: center;color: white;'>E-ONLINE AUCTION RECOMMEDATION SYSTEM</h1>", unsafe_allow_html=True)
#    image1_url = "https://uxdt.nic.in/wp-content/uploads/2021/05/gepnic-gepnic-logo-02-01.jpg?x93453"
#    st.image(image1_url, use_column_width=True, width=300)
#    st.markdown("<p style='text-align: justify;'>National Informatics Center (NIC), Ministry of Electronics and Information Technology, Government of India has developed eProcurement software system, in GePNIC to cater to the procurement/tendering requirements of the government departments and organizations. GePNIC was launched in 2007 and has matured as a product over the decade. The system is generic in nature and can easily be adopted for all kinds of procurement activities such as Goods, Services and Works by across Government.Government eProcurement System of NIC (GePNIC) is an online solution to conduct all stages of a procurement process.GePNIC converts tedious procurement process into an economical, transparent and more secure system. Today, many Government organizations and public sector units have adopted GePNIC. Understanding the valuable benefits and advantages of this eProcurement system, Government of India, has included this under one of the Mission Mode Projects of National eGovernance Plan (NeGP) for implementation in all the Government departments across the country.</p>", unsafe_allow_html=True) 
# elif selected_option == "ABOUT":
# #    st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://img.freepik.com/free-icon/legal-hammer-symbol_318-64606.jpg?size=626&ext=jpg&uid=R101051765&ga=GA1.2.1088004906.1682729121&semt=robertav1_2_sidr' width='50' style='margin-right: 10px;'> <h2 style='color: #424242;'>E-ONLINE AUCTION RECOMMENDATION</h2></div>", unsafe_allow_html=True)
# #    st.markdown("<marquee style='width: 80%; color: red;'><p>HURRY!!! It's Time to get the pace</p></marquee>", unsafe_allow_html=True)
   
#    st.markdown("<h1 style='text-align: center;'>INTRODUCTION TO AUCTION RECOMMENDATION SYSTEM</h1>", unsafe_allow_html=True)
#    image_url = "https://img.freepik.com/premium-vector/auction-hammer-icon-comic-style-court-sign-cartoon-vector-illustration-white-isolated-background-tribunal-splash-effect-business-concept_157943-2427.jpg?w=2000"
#    st.image(image_url)
#    st.markdown("<p style='text-align: justify;'>This Project presents an Auction Recommendation System based on machine learning techniques. Online auctions have become increasingly popular in recent years as a means of buying and selling goods and services. With the rise of e-commerce platforms and the growth of online marketplaces, auction systems have become more sophisticated and complex. One of the challenges in this domain is to provide users with personalized recommendations of auctions that match their preferences and needs. To address this challenge, machine learning techniques have been applied to develop auction recommendation systems.The proposed system utilizes GepNIC auction data to predict the most suitable auction for a bidder based on the bidder location. The system employs Hybrid Methodology (collaborative filtering algorithms, content-based filtering, Knowledge based filtering, Demographic based filtering) techniques to generate personalized auction recommendations for each bidder. The model uses features such as tender_type_name ,tender_category_name tender_form_contract_name to make accurate predictions. In this paper, we propose an auction recommendation system based on the k-nearest neighbors (KNN) algorithm. The KNN algorithm is a widely used machine learning technique that is often used in recommendation systems. The model is trained on a large dataset of GepNIC auction data, which allows it to capture patterns and trends from the Bidder. The system is designed to be scalable and can handle large volumes of data, making it suitable for use in large-scale e-commerce platforms. The system incorporates user feedback to continuously improve the recommendations and adapt to changing user preferences. The system is evaluated using a real-world auction dataset and achieves high accuracy in predicting the most appropriate auction for an given bidder details. The evaluation results show that the proposed system outperforms other existing recommendation approaches in terms of accuracy and coverage.The proposed auction recommendation system is integrated into website to enhance the user experience and increase engagement. The proposed auction recommendation system has several advantages over existing systems. First, it is designed to be scalable and can handle large volumes of data, making it suitable for use in large-scale e-commerce platforms. Second, the system uses a hybrid approach that combines both collaborative and content-based filtering to provide more accurate recommendations. Finally, the system incorporates user feedback to continuously improve the recommendations and adapt to changing user preferences.</p>", unsafe_allow_html=True) 
# elif selected_option == "DATASET":
# #    st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://img.freepik.com/free-icon/legal-hammer-symbol_318-64606.jpg?size=626&ext=jpg&uid=R101051765&ga=GA1.2.1088004906.1682729121&semt=robertav1_2_sidr' width='50' style='margin-right: 10px;'> <h2 style='color: #424242;'>E-ONLINE AUCTION RECOMMENDATION</h2></div>", unsafe_allow_html=True)
# #    st.markdown("<marquee style='width: 80%; color: red;'><p>HURRY!!! It's Time to get the pace</p></marquee>", unsafe_allow_html=True)
   
#    st.markdown("<h1 style='text-align: center;'>ABOUT THE DATASET</h1>", unsafe_allow_html=True)
# #    image1_url = Image.open("C:/Users/maadh/Pictures/Screenshots/Screenshot (122).png")
# #    st.image(image1_url, use_column_width=True, width=300)
#    text = """
#    <p style='text-align: justify;'>
#    <b>Here are some key points about the auction dataset:</b>
#    <ul>
#    <li>tender_id: The unique identifier for each tender.</li>
#    <li>bidder_id: The unique identifier for each bidder.</li>
#    <li>tender_type_name: The type of tender, such as "Open Tender", "Limited", "Global Tenders", etc.</li>
#    <li>tender_category_name: The category of the tender, such as "Works", "Goods", or "Services".</li>
#    <li>tender_form_contract_name: The type of contract for the tender, such as "Item Rate", "Percentage", "Fixed-rate", etc.</li>
#    <li>bidder_name: The name of the bidder who participated in the tender.</li>
#    <li>bid_Value: The value of the bid submitted by the bidder.</li>
#    <li>bidder_location: The location of the bidder, such as "Delhi", "Tamil Nadu", "Rajasthan", etc.</li>
#    <li>bidder_ratings: The rating of the bidder based on their past performance.</li>
#    <li>tender_location: The location where the tender is being conducted.</li>
#    </ul>
#    </p>
#     """

#    st.markdown(text, unsafe_allow_html=True)
#    st.markdown("""
# - The `bidder_location` attribute contains the locations of bidders who participated in the auction, which include "Delhi", "Tamil Nadu", "Rajasthan", "Kerala", "Andhra Pradesh", "Bihar", "Punjab", "Telangana", "Gujarat", and "Madhya Pradesh".
# - The `tender_type_name` attribute contains the different types of tenders, such as "Open Tender", "Limited", "Open Limited", "Global Tenders", "EOI", and "Test".
# - The `tender_category_name` attribute contains the categories of the tender, such as "Works", "Goods", and "Services".
# - The `tender_form_contract_name` attribute contains the different types of contracts for the tender, such as "Item Rate", "Percentage", "Item Wise", "Supply", "Lump-sum", "Supply and Service", "Service", "Fixed-rate", "Tender cum Auction", "Turn-key", and "Piece-work".
# """, unsafe_allow_html=True)
#    st.markdown("**HERE IS THE AUCTION DATASET**", unsafe_allow_html=True)
#    auction = pd.read_csv('C:/Users/maadh/Documents/New Auction/2023_GePNIC dataset.csv')
#    st.write(auction)
#    csv = auction.to_csv(index=False)
#    b64 = base64.b64encode(csv.encode()).decode()
#    href = f'<a href="data:file/csv;base64,{b64}" download="your_dataset.csv">Download CSV file</a>'
#    st.markdown(href, unsafe_allow_html=True)
# elif selected_option == "CONTACT":
# #    st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://img.freepik.com/free-icon/legal-hammer-symbol_318-64606.jpg?size=626&ext=jpg&uid=R101051765&ga=GA1.2.1088004906.1682729121&semt=robertav1_2_sidr' width='50' style='margin-right: 10px;'> <h2 style='color: #424242;'>E-ONLINE AUCTION RECOMMENDATION</h2></div>", unsafe_allow_html=True)
# #    st.markdown("<marquee style='width: 80%; color: red;'><p>HURRY!!! It's Time to get the pace</p></marquee>", unsafe_allow_html=True)
#    st.markdown("<h1 style='text-align: justify;'>R.MADHURI DEVI</h1>", unsafe_allow_html=True)
#    st.markdown("<p style='text-align: justify;'>Phone No: 7904079612</p>", unsafe_allow_html=True)
#    st.markdown("<p style='text-align: justify;'>Email Id: maadhudevi123@gmail.com</p>", unsafe_allow_html=True)
# elif selected_option == "FEEDBACK":
    
#     st.markdown("<h1 style='text-align: justify;'>WEBSITE FEEDBACK FORM </h1>", unsafe_allow_html=True)
    
#     # Define the filename for the Excel file
#     # output_folder = "output"
#     # if not os.path.exists(output_folder):
#     #     os.makedirs(output_folder)
#     # filename = os.path.join(output_folder, "feedback.xlsx")
#     filename = os.path.join(os.path.expanduser("~"), "Documents", "feedback.xlsx")

#     feedback_data = {
#         'Name': '',
#         'Email': '',
#         'Is the Website user friendly to use:'
#         'Feedback': '',
#         'Rating': 0
#     }

#     with st.form(key='feedback_form'):
#         name = st.text_input(label='Name')
#         email = st.text_input(label='Email')
#         credits = st.radio(label='Does the website userfriendly?', options=['Yes', 'No'])
#         feedback = st.text_area(label='Feedback')
#         rating = st.slider('Rate your experience', 1, 5)
#         submitted = st.form_submit_button(label='Submit')
#         if submitted:
#             # Update the feedback data with the user's inputs
#             # feedback_data['Name'] = name
#             # feedback_data['Email'] = email
#             # feedback_data['Does the website userfriendly?'] = credits
#             # feedback_data['Feedback'] = feedback
#             # feedback_data['Rating'] = rating

#             # # Convert the feedback data into a Pandas DataFrame
#             # feedback_df = pd.DataFrame([feedback_data])
#             # # Save the feedback DataFrame into an Excel file
#             # feedback_df.to_excel(filename, index=False)
#             st.success('Thank you for your feedback!')
# st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://img.freepik.com/free-icon/legal-hammer-symbol_318-64606.jpg?size=626&ext=jpg&uid=R101051765&ga=GA1.2.1088004906.1682729121&semt=robertav1_2_sidr' width='50' style='margin-right: 10px;'> <h2 style='color: #424242;'>E-AUCTION RECOMMENDATION PORTAL</h2></div>", unsafe_allow_html=True)
# st.markdown("<marquee style='width: 80%; color: red;'><p>HURRY!!! It's Time to get the pace</p></marquee>", unsafe_allow_html=True)

# Load the dataset
df = pd.read_csv('C:/Users/maadh/Downloads/2023_GePNIC dataset.csv')
df.head()

df['auction_description'] = df['tender_type_name'] + ', ' + df['tender_category_name'] + ', ' + df['tender_form_contract_name'] + ', ' + df['tender_location']
df.head()

st.subheader('Enter the bidder information')
# Get user input
bidder_location_options= ['Delhi','Tamil Nadu' ,'Rajasthan' ,'Kerala' ,'Andhra Pradesh', 'Bihar',
 'Punjab', 'Telangana', 'Gujarat', 'Madhya Pradesh'] 
bidder_location = st.selectbox('Enter bidder_location',bidder_location_options)

# Filter the auction data based on the bidder_location
df = df[df['tender_location'] == bidder_location]

tender_type_options = ['Open Tender' ,'Limited' ,'Open Limited', 'Global Tenders' ,'EOI' ,'Test']  # Replace with actual list of options
tender_type_name = st.selectbox('Enter tender_type_name',tender_type_options)
tender_category_name_options= ['Works' ,'Goods', 'Services']
tender_category_name = st.selectbox('Enter tender_category_name',tender_category_name_options)
tender_form_contract_name_options = ['Item Rate' ,'Percentage', 'Item Wise' ,'Supply' ,'Lump-sum',
 'Supply and Service', 'Service' ,'Fixed-rate', 'Tender cum Auction',
 'Turn-key' ,'Piece-work']
tender_form_contract_name = st.selectbox('Enter tender_form_contract_name',tender_form_contract_name_options)
user = tender_type_name + ',' + tender_category_name + ',' + tender_form_contract_name + ',' + bidder_location

# Load a pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Tokenize the user input and dataset

user_tokens = tokenizer(user, padding=True, truncation=True, return_tensors='pt')
dataset_tokens = tokenizer(list(df['auction_description']), padding=True, truncation=True, return_tensors='pt')

# Use the pre-trained Roberta model to extract features from the tokenized text
with torch.no_grad():
    user_features = model(user_tokens['input_ids'], attention_mask=user_tokens['attention_mask'])[0][:, 0, :]
    dataset_features = model(dataset_tokens['input_ids'], attention_mask=dataset_tokens['attention_mask'])[0][:, 0, :]

# Calculate the cosine similarity between the user input and each row in the dataset
cosine_similarities = cosine_similarity(user_features.numpy(), dataset_features.numpy()).flatten() 
cosine_similarities

# Get the indices of the top 10 most similar rows using content-based filtering
content_based_indices = cosine_similarities.argsort()[:-11:-1]
content_based_indices

# Train a KNN model using the dataset
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(dataset_features.numpy())

# Find the top 10 nearest neighbors using collaborative filtering
collaborative_indices = knn_model.kneighbors(user_features.numpy(), n_neighbors=10, return_distance=False)[0]
collaborative_indices

# Combine the indices from content-based filtering and collaborative filtering
hybrid_indices = np.unique(np.concatenate((content_based_indices, collaborative_indices), axis=None))
hybrid_indices

# Take the columns with index values and print them
print('Top 10 most similar rows:')
for index in hybrid_indices:
    recommend = np.take(df.iloc[index].values, [0, 2, 3, 4, -2, 10])
    print(recommend)




