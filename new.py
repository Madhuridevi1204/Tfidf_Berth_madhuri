import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
import base64
import os
from PIL import Image
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
st.sidebar.markdown("<div style='display: flex; align-items: center; justify-content: left;'><img src='https://icon-library.com/images/home-menu-icon/home-menu-icon-7.jpg' width='50'> <h1 style='color: #424242;'><b>MAIN MENU</b></h1></div>", unsafe_allow_html=True)

# st.sidebar.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='' width='50' style='margin-right: 10px;'> <h1 style='color: #424242;'>MAIN MENU</h1></div>", unsafe_allow_html=True)

options = ["RECOMMEND AUCTION", "HOME", "ABOUT", "DATASET", "CONTACT", "FEEDBACK"]
selected_option = st.sidebar.radio("", options)
if selected_option == "RECOMMEND AUCTION":
   st.write("")
   
elif selected_option == "HOME":
#    st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://img.freepik.com/free-icon/legal-hammer-symbol_318-64606.jpg?size=626&ext=jpg&uid=R101051765&ga=GA1.2.1088004906.1682729121&semt=robertav1_2_sidr' width='50' style='margin-right: 10px;'> <h2 style='color: #424242;'>E-ONLINE AUCTION RECOMMENDATION</h2></div>", unsafe_allow_html=True)
#    st.markdown("<marquee style='width: 80%; color: red;'><p>HURRY!!! It's Time to get the pace</p></marquee>", unsafe_allow_html=True)
   st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><h1 style='color: #424242;'> GEPNIC E-PROCUREMENT</h1></div>", unsafe_allow_html=True)
   st.markdown("<h1 style='text-align: center;color: white;'>E-ONLINE AUCTION RECOMMEDATION SYSTEM</h1>", unsafe_allow_html=True)
   image1_url = "https://uxdt.nic.in/wp-content/uploads/2021/05/gepnic-gepnic-logo-02-01.jpg?x93453"
   st.image(image1_url, use_column_width=True, width=300)
   st.markdown("<p style='text-align: justify;'>National Informatics Center (NIC), Ministry of Electronics and Information Technology, Government of India has developed eProcurement software system, in GePNIC to cater to the procurement/tendering requirements of the government departments and organizations. GePNIC was launched in 2007 and has matured as a product over the decade. The system is generic in nature and can easily be adopted for all kinds of procurement activities such as Goods, Services and Works by across Government.Government eProcurement System of NIC (GePNIC) is an online solution to conduct all stages of a procurement process.GePNIC converts tedious procurement process into an economical, transparent and more secure system. Today, many Government organizations and public sector units have adopted GePNIC. Understanding the valuable benefits and advantages of this eProcurement system, Government of India, has included this under one of the Mission Mode Projects of National eGovernance Plan (NeGP) for implementation in all the Government departments across the country.</p>", unsafe_allow_html=True) 
elif selected_option == "ABOUT":
#    st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://img.freepik.com/free-icon/legal-hammer-symbol_318-64606.jpg?size=626&ext=jpg&uid=R101051765&ga=GA1.2.1088004906.1682729121&semt=robertav1_2_sidr' width='50' style='margin-right: 10px;'> <h2 style='color: #424242;'>E-ONLINE AUCTION RECOMMENDATION</h2></div>", unsafe_allow_html=True)
#    st.markdown("<marquee style='width: 80%; color: red;'><p>HURRY!!! It's Time to get the pace</p></marquee>", unsafe_allow_html=True)
   
   st.markdown("<h1 style='text-align: center;'>INTRODUCTION TO AUCTION RECOMMENDATION SYSTEM</h1>", unsafe_allow_html=True)
   image_url = "https://img.freepik.com/premium-vector/auction-hammer-icon-comic-style-court-sign-cartoon-vector-illustration-white-isolated-background-tribunal-splash-effect-business-concept_157943-2427.jpg?w=2000"
   st.image(image_url)
   st.markdown("<p style='text-align: justify;'>This Project presents an Auction Recommendation System based on machine learning techniques. Online auctions have become increasingly popular in recent years as a means of buying and selling goods and services. With the rise of e-commerce platforms and the growth of online marketplaces, auction systems have become more sophisticated and complex. One of the challenges in this domain is to provide users with personalized recommendations of auctions that match their preferences and needs. To address this challenge, machine learning techniques have been applied to develop auction recommendation systems.The proposed system utilizes GepNIC auction data to predict the most suitable auction for a bidder based on the bidder location. The system employs Hybrid Methodology (collaborative filtering algorithms, content-based filtering, Knowledge based filtering, Demographic based filtering) techniques to generate personalized auction recommendations for each bidder. The model uses features such as tender_type_name ,tender_category_name tender_form_contract_name to make accurate predictions. In this paper, we propose an auction recommendation system based on the k-nearest neighbors (KNN) algorithm. The KNN algorithm is a widely used machine learning technique that is often used in recommendation systems. The model is trained on a large dataset of GepNIC auction data, which allows it to capture patterns and trends from the Bidder. The system is designed to be scalable and can handle large volumes of data, making it suitable for use in large-scale e-commerce platforms. The system incorporates user feedback to continuously improve the recommendations and adapt to changing user preferences. The system is evaluated using a real-world auction dataset and achieves high accuracy in predicting the most appropriate auction for an given bidder details. The evaluation results show that the proposed system outperforms other existing recommendation approaches in terms of accuracy and coverage.The proposed auction recommendation system is integrated into website to enhance the user experience and increase engagement. The proposed auction recommendation system has several advantages over existing systems. First, it is designed to be scalable and can handle large volumes of data, making it suitable for use in large-scale e-commerce platforms. Second, the system uses a hybrid approach that combines both collaborative and content-based filtering to provide more accurate recommendations. Finally, the system incorporates user feedback to continuously improve the recommendations and adapt to changing user preferences.</p>", unsafe_allow_html=True) 
elif selected_option == "DATASET":
#    st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://img.freepik.com/free-icon/legal-hammer-symbol_318-64606.jpg?size=626&ext=jpg&uid=R101051765&ga=GA1.2.1088004906.1682729121&semt=robertav1_2_sidr' width='50' style='margin-right: 10px;'> <h2 style='color: #424242;'>E-ONLINE AUCTION RECOMMENDATION</h2></div>", unsafe_allow_html=True)
#    st.markdown("<marquee style='width: 80%; color: red;'><p>HURRY!!! It's Time to get the pace</p></marquee>", unsafe_allow_html=True)
   
   st.markdown("<h1 style='text-align: center;'>ABOUT THE DATASET</h1>", unsafe_allow_html=True)
#    image1_url = Image.open("C:/Users/maadh/Pictures/Screenshots/Screenshot (122).png")
#    st.image(image1_url, use_column_width=True, width=300)
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
   auction = pd.read_csv('C:/Users/maadh/Documents/New Auction/2023_GePNIC dataset.csv')
   st.write(auction)
   csv = auction.to_csv(index=False)
   b64 = base64.b64encode(csv.encode()).decode()
   href = f'<a href="data:file/csv;base64,{b64}" download="your_dataset.csv">Download CSV file</a>'
   st.markdown(href, unsafe_allow_html=True)
elif selected_option == "CONTACT":
#    st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://img.freepik.com/free-icon/legal-hammer-symbol_318-64606.jpg?size=626&ext=jpg&uid=R101051765&ga=GA1.2.1088004906.1682729121&semt=robertav1_2_sidr' width='50' style='margin-right: 10px;'> <h2 style='color: #424242;'>E-ONLINE AUCTION RECOMMENDATION</h2></div>", unsafe_allow_html=True)
#    st.markdown("<marquee style='width: 80%; color: red;'><p>HURRY!!! It's Time to get the pace</p></marquee>", unsafe_allow_html=True)
   st.markdown("<h1 style='text-align: justify;'>R.MADHURI DEVI</h1>", unsafe_allow_html=True)
   st.markdown("<p style='text-align: justify;'>Phone No: 7904079612</p>", unsafe_allow_html=True)
   st.markdown("<p style='text-align: justify;'>Email Id: maadhudevi123@gmail.com</p>", unsafe_allow_html=True)
elif selected_option == "FEEDBACK":
    
    st.markdown("<h1 style='text-align: justify;'>WEBSITE FEEDBACK FORM </h1>", unsafe_allow_html=True)


    # Set the default file path
    filename = os.path.join(os.path.expanduser("~"), "Documents", "feedback.xlsx")

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
            
            # Check if the feedback file exists
            if os.path.isfile(filename):
                # Read the existing feedback data into a DataFrame
                existing_df = pd.read_excel(filename)
                # Concatenate the existing and new feedback DataFrames
                feedback_df = pd.concat([existing_df, pd.DataFrame([feedback_data])], ignore_index=True)
            else:
                # If the feedback file does not exist, create a new DataFrame with the new feedback data
                feedback_df = pd.DataFrame([feedback_data])
            
            # Save the feedback DataFrame into the Excel file, overwriting the previous data
            feedback_df.to_excel(filename, index=False)
            st.success('Thank you for your feedback!')


st.markdown("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://img.freepik.com/free-icon/legal-hammer-symbol_318-64606.jpg?size=626&ext=jpg&uid=R101051765&ga=GA1.2.1088004906.1682729121&semt=robertav1_2_sidr' width='50' style='margin-right: 10px;'> <h2 style='color: #424242;'>E-AUCTION RECOMMENDATION PORTAL</h2></div>", unsafe_allow_html=True)
st.markdown("<marquee style='width: 80%; color: red;'><p>HURRY!!! It's Time to get the pace</p></marquee>", unsafe_allow_html=True)


# Load the auction data


# url = 'https://raw.githubusercontent.com/Madhuridevi1204/new_auction/blob/main/2023_GePNIC%20dataset.csv'
auction = pd.read_csv('C:/Users/maadh/Documents/New Auction/2023_GePNIC dataset.csv')

# Data Visualisation

# Display the total count of states using pie chart 
state_counts = auction['tender_location'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(state_counts, labels=state_counts.index, autopct='%1.1f%%')
ax1.set_title('Number of Auctions by State')
# st.pyplot(fig1)

# Display the distribution of the data across tender_location
fig2, ax2 = plt.subplots()
sns.countplot(x='tender_location', data=auction, ax=ax2)
ax2.set_title('Distribution of Tender Location')
ax2.tick_params(axis='x', rotation=90)
# st.pyplot(fig2)

# Display the distribution of the data across bidder_location
fig3, ax3 = plt.subplots()
sns.countplot(x='bidder_location', data=auction, ax=ax3)
ax3.set_title('Distribution of Bidder Location')
ax3.tick_params(axis='x', rotation=90)
# st.pyplot(fig3)
# Display options for the plots
options = ["None", "Auctions by State", "Tender Location Distribution", "Bidder Location Distribution"]
selectbox_style = "<style>div[role='listbox'] ul { font-size: 22px !important; font-weight: bold !important; }</style>"
st.markdown(selectbox_style, unsafe_allow_html=True)
plot_options = st.sidebar.selectbox('STATISTICAL INFORMATION PLOTS', options)

if 'Auctions by State' in plot_options:
    st.pyplot(fig1)

if 'Tender Location Distribution' in plot_options:
    st.pyplot(fig2)

if 'Bidder Location Distribution' in plot_options:
    st.pyplot(fig3)


# Create the sidebar
# st.sidebar.title('Auction Filters')
# tender_type_name = st.sidebar.text_input('Tender Type Name')
# tender_category_name = st.sidebar.text_input('Tender Category Name')
# tender_form_contract_name = st.sidebar.text_input('Tender Form Contract Name')

# Concatenate all text fields into a single column
auction['auction_description'] = auction['tender_type_name'] + ',' + auction['tender_category_name'] + ',' + auction['tender_form_contract_name'] + ',' + auction['tender_location']

st.subheader('Enter the bidder information')
# Get user input
bidder_location_options= ['Delhi','Tamil Nadu' ,'Rajasthan' ,'Kerala' ,'Andhra Pradesh', 'Bihar',
 'Punjab', 'Telangana', 'Gujarat', 'Madhya Pradesh'] 
bidder_location = st.selectbox('Enter bidder_location',bidder_location_options)

# Filter the auction data based on the bidder_location
auction = auction[auction['tender_location'] == bidder_location]

# Display the filtered data
# st.write('Filtered Data:', auction)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
vectorizer1 = vectorizer.fit_transform(auction['auction_description'])

# Display the results
# st.write('Filtered Data:', auction)
# st.write('Vectorized Data:', vectorizer1)

# Compute cosine similarity between all pairs of documents
cosine_sim = cosine_similarity(vectorizer1)

# Train a k-NN model on the user-item matrix for collaborative filtering
# k = st.sidebar.slider('Select the number of neighbors to consider:', 1, 10, 5)
k=5
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k)
model_knn.fit(vectorizer1)

# Get user input
# Create the sidebar
tender_type_options = ['Open Tender' ,'Limited' ,'Open Limited', 'Global Tenders' ,'EOI' ,'Test']  # Replace with actual list of options
tender_type_name = st.selectbox('Enter tender_type_name',tender_type_options)
tender_category_name_options= ['Works' ,'Goods', 'Services']
tender_category_name = st.selectbox('Enter tender_category_name',tender_category_name_options)
tender_form_contract_name_options = ['Item Rate' ,'Percentage', 'Item Wise' ,'Supply' ,'Lump-sum',
 'Supply and Service', 'Service' ,'Fixed-rate', 'Tender cum Auction',
 'Turn-key' ,'Piece-work']
tender_form_contract_name = st.selectbox('Enter tender_form_contract_name',tender_form_contract_name_options)
user_data = tender_type_name + ',' + tender_category_name + ',' + tender_form_contract_name + ',' + bidder_location

# Add user input to the DataFrame
user_row = pd.DataFrame({'auction_description': [user_data]})
auction = pd.concat([auction, user_row], axis=0)

# Add user input to the DataFrame
user_row = pd.DataFrame({'auction_description': [user_data]})
auction = pd.concat([auction, user_row], axis=0)


# Vectorize the user input using the same vectorizer object as before
vectorizer2 = vectorizer.transform([user_data])


# Compute cosine similarity between user input and all other auctions
cosine_sim_user = cosine_similarity(vectorizer2, vectorizer1)

# Find index of user input in the DataFrame
user_index = auction.index[-1]

# Get cosine similarities between user input and all other auctions
similarities = cosine_sim_user[0]


# Sort cosine similarities in descending order and get the top n most similar auctions
n = st.sidebar.slider('Select the number of top recommendations to generate:', 1, 20, 10) # Number of top recommendations to generate
top_n_indices_content = similarities.argsort()[::-1][:n]
top_n_auctions_content = auction.iloc[top_n_indices_content]

# Get top n recommendations using collaborative filtering
_, indices = model_knn.kneighbors(vectorizer2, n_neighbors=n)
top_n_indices_collab = indices[0]
top_n_auctions_collab = auction.iloc[top_n_indices_collab]

# Combine content-based and collaborative recommendations
top_n_auctions = pd.concat([top_n_auctions_content, top_n_auctions_collab]).drop_duplicates()

# Display top n recommended auctions to the user
if st.button("Recommend Auctions"):
    st.header("Top Recommended Auctions")
    table = []
    for i, row in top_n_auctions.iterrows():
        table.append([i, row['auction_description'], row['tender_id'], row['bidder_name']])
    st.table(pd.DataFrame(table, columns=["Index", "auction_description", "Tender ID", "Bidder Name"]))

    
# Plot the correlation matrix as a heatmap

fig4, ax4 = plt.subplots()
corr_matrix = auction.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, ax=ax4)
ax4.set_title('Correlation Plot for Auction Data')
# st.pyplot(fig4)

# Displays the distribution of the number of bids per auction using seaborn and matplotlib
bids_per_auction = auction.groupby('tender_id').size().reset_index(name='num_bids')
fig5, ax5 = plt.subplots()
sns.histplot(data=bids_per_auction, x='num_bids', kde=True, ax=ax5)
ax5.set_title('Distribution of Bids per Auction')
# st.pyplot(fig5)

# Displays the distribution of tender type, tender category, and tender form contract name in the auction dataset.
fig6, ax6 = plt.subplots()
sns.countplot(x='tender_type_name', data=auction, ax=ax6)
ax6.set_title('Distribution of Tender Type')
# st.pyplot(fig6)

fig7, ax7 = plt.subplots()
sns.countplot(x='tender_category_name', data=auction, ax=ax7)
ax7.set_title('Distribution of Tender Category')
# st.pyplot(fig7)

fig8, ax8 = plt.subplots()
sns.countplot(x='tender_form_contract_name', data=auction, ax=ax8)
ax8.set_title('Distribution of Tender Form Contract')
ax8.tick_params(axis='x', rotation=45)

# Plot the distribution of auction categories in the filtered data
# st.subheader('Distribution of Auction Categories')
# fig9, ax9 = plt.subplots()
# sns.countplot(x='tender_category_name', data=auction, ax=ax9)
# plt.xticks(rotation=45)



options = ["None",  "Correlation Plot", "Bids per Auction Distribution", "Tender Type Distribution", "Tender Category Distribution", "Tender Form Contract Distribution","Distribution of Auction Categories"]
selectbox_style = "<style>div[role='listbox'] ul { font-size: 22px !important; font-weight: bold !important; }</style>"
st.markdown(selectbox_style, unsafe_allow_html=True)
plot_options = st.sidebar.selectbox('PLOT VARAITIONS', options)


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
    sns.countplot(x='tender_category_name', data=auction, ax=ax9)
    plt.xticks(rotation=45)
    st.pyplot(fig9)



