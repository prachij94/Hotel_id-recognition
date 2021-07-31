# import required libraries
import streamlit as st 
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss
import pickle

# import required libraries for visualization of clusters and nearest centroids to test data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
sns.set()


#set the title of the page
st.title("Hotel prediction to combat Human Trafficking")


def predict(X):
    '''
    This function takes the test image as input
    and returns the predicted hotel IDs for it with their indices
    '''
    
    #read image array as a tensor
    image = tf.convert_to_tensor(X,dtype=np.uint8)
    
    
    #resize image
    image = tf.image.resize(image, [64,64])
    
    #convert to float32
    image = tf.cast(image, tf.float32)
    
    # normalize image to [0,1] range
    image /= 255.0
    
    tf.keras.backend.clear_session()
    
    
    #load the saved test model
    test_model = tf.keras.models.load_model('test_model_saved', compile=False)
    
    # get embeddings for the test image
    test_embeds = test_model(np.expand_dims(image,0))
    
    #normalize the image pixels
    norm = np.linalg.norm(test_embeds, axis=1, keepdims=True)
    test_embeds = test_embeds / norm
    
    #convert from tensor to array
    test_embeds = test_embeds.numpy()
    
    
    #load the saved kmeans trained model index
    kmeans_index = faiss.read_index("kmeans_trained.index")
    
    #search for nearest distance and indices of the test embeddings to the centroids
    distances, indices = kmeans_index.search(test_embeds,5)
    
    # load the saved label encoder
    with open('le.pkl', "rb") as input_file:
        le = pickle.load(input_file)
    
    # get the actual outputs using the inverse transform of the saved label encoder
    actual_output_labels = le.inverse_transform(indices[0])
    
    return list(actual_output_labels),indices
    
def visualize(labels,indices):
    '''
    This function takes the predicted hotel ID and indices for the input image
    and return a plot having the train embeddings with all unique the hotel IDs as centroids
    and also has marked the predicted hotel IDs for the input image
    '''
    
    # load the train embeddings
    train_embeds = np.load('train_embeddings.npy')
    
    #load the saved centroids
    with open('kmeans_centroids.pkl', "rb") as input_file:
        kmeans_centroids= pickle.load(input_file)
    
    # reduce the high dimensional data to 2D for visualization using PCA
    pca1 = PCA(2)
    pca2 = PCA(2)
     
    #Transform the data
    centroids = pca1.fit_transform(kmeans_centroids)
    train_data = pca2.fit_transform(train_embeds)
    
    #read the train data csv
    train_df = pd.read_csv('train.csv')

    
    #create the figure size
    fig, ax = plt.subplots()
    fig.set_size_inches(25, 15)
    
    #get the unique original hotel id's from the train csv
    u_labels = np.unique(train_df['hotel_id'])
     
    
    #plotting the results:
    
    #iterate over the unique hotel id's
    for i in u_labels:
        
        #plot the images of each unique hotel id in a separate colour
        plt.scatter(train_data[train_df['hotel_id']==i,0] , train_data[train_df['hotel_id']==i,1], cmap='viridis', s=50)
        
    #plot the centroids of each of the above cluster
    plt.scatter(centroids[:,0] , centroids[:,1], s = 5, color = 'black',alpha=0.8,label='Hotel ID centroid')
    
    # define colours for marking the 5 nearest neighbours
    predicted_colours = ['tomato','gold','springgreen','cornflowerblue','hotpink']
    count=0
    
    #get the max dimensions of the x and y axis
    left_x, right_x = plt.xlim()
    min_y, max_y = plt.ylim()
    
    #iterate over the 5 nearest predicted centroid indices for the test embeddings
    for j in list(indices[0]):
        
        #plot the nearest centroid i.e. predicted hotel ID for the test input image
        plt.scatter(centroids[j,0] , centroids[j,1], s=1500, c=predicted_colours[count], alpha=0.9,label='Predicted Hotel ID '+str(count+1),marker='*',edgecolors='white',linewidth=2)
        
        # also show the predicted hotel ID on the marked point
        plt.text(0.05+left_x+(0.05*count), min_y+(0.01),str(labels[count]),fontsize=15,bbox=dict(edgecolor='black',facecolor=predicted_colours[count], alpha=0.9))
        
        count+=1
        
    #add legend to mark labels
    plt.legend(fontsize='large')
    
    #return the generated plot
    return fig

#Show the image uploader widget
uploaded_file = st.file_uploader("Upload a hotel image...", type="jpg")

#enter the loop if a valid file is uploaded
if uploaded_file is not None:
    
    #open the uploaded image
    image = Image.open(uploaded_file)
    
    #display the uploaded image
    st.image(image, caption='Uploaded Hotel Image', use_column_width=True)
    
    st.write("")
    
    #show text for showing progress
    placeholder1 = st.empty()
    placeholder1.markdown("Classifying with the uploaded image to get similar Hotels...")
    
    # call the predict() to get the similar hotel ID predictions and their indices
    labels,indices = predict(image)
    
    #Show text that results have been predicted
    placeholder1.markdown("<font color=‘green’>Matches found!</font>", unsafe_allow_html=True)
    
    #Display the results
    st.write('Closest match: Hotel ID ',labels[0])
    st.write("")
    st.write('Other most similar Hotel IDs are (most relevant first) :')
    st.write(labels[1],'     ',labels[2],'     ',labels[3],'     ',labels[4])
    st.write('These Hotel IDs can also be looked up in the TraffickCam dataset to get the related hotel chain')
    st.write("")
    st.write("")
    
    #Show a button to ask the user if they want to see cluster plot
    start_execution = st.button('Click to Visualize')
    st.write("")
    st.write("")
    
    # enter the loop when button is clicked
    if start_execution:
        
        #display text for progress
        placeholder2 = st.empty()
        placeholder2.markdown("This may take some while. Please wait for a few minutes...")
        st.write("")
        
        #show the loading gif till the plot is getting created using visualize()
        gif_runner = st.image('loading1.gif')
        fig =  visualize(labels,indices)
        
        #when the plot has been returned from the function call, removing the loading gif
        placeholder2.markdown("This is how the uploaded image compares to other hotels and resembles nearest to the marked hotels :")
        gif_runner.empty()
        
        #display the plotted graph
        st.pyplot(fig)