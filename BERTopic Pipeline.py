# Torch is a module that provides a wide range of functionalities for deep learning and tensor computations
import torch
print(f"torch version: {torch.__version__}")

import transformers # a library for NLP tasks, providing pre-trained models and tokenizers
print(f"transformers version: {transformers.__version__}")

#AutoTokenizer is a class for tokenizing text using pre-trained models
#AutoModel is a class for laoding pre-trained models for different NLP tasks
from transformers import AutoTokenizer, AutoModel

#BERTopic-a topic modeling library that uses transformers for embeddings
from bertopic import BERTopic
#SentenceTransformer is a class for generating sentence embeddings using pre-trained models
from sentence_transformers import SentenceTransformer #embedding
#Wrap the BERT model with SentenceTransformer's transformer and pooling layers
from transformers.utils import cached_file

#Setting the logging level for 'transformers' - a setting that determines the severity of messages that are logged by the library.
transformers.logging.set_verbosity_error()

#Importing additional libraries for data handling and visualization
import pandas as pd #Pandas library is used for data manipulation and analysis
import numpy as np #Numpy library is used for numerical computations and array manipulations
#An Array is a data structure that stores a collection of elements, typically of the same type unlike a list   
from wordcloud import WordCloud #Word clouds are visual representations of word frequency in a text corpus
import matplotlib.pyplot as plt #Matplotlib is a plotting library for creating static, animated, and interactive visualizations in Python
import math #Math library provides mathematical functions and constants. Why is this library imported here? c-TFIDF calculations may require mathematical operations involving logarithms, square roots, etc.
from matplotlib.backends.backend_pdf import PdfPages #Pdfpages is a backend class for saving multiple plots to a single PDF file
import os

#from scikit-learn import CountVectorizer
#CountVectorizer is a class in scikit-learn is used to convert a collection of text documents to a matrix of token counts
#CountVectorizer is used to create a document-term matrix, which is necessary for calculating the c-TFIDF (Class-based Term Frequency-Inverse Document Frequency) representation
#c-TFIDF is a variant of TF-IDF that takes into account the class labels of the documents, allowing for better topic representation
from sklearn.feature_extraction.text import CountVectorizer

#ClassTfidfTransformer is a class in BERTopic that applies c-TFIDF transformation to the document-term matrix   
#It is used to compute the term frequency-inverse document frequency (TF-IDF) representation of the text data
#What is TF-IDF? TF-IDF is a statistical measure that evaluates the importance of a word in a document relative to a collection of documents (corpus)
# It is often used in information retrieval and text mining to identify relevant documents for a given query.
from bertopic.vectorizers import ClassTfidfTransformer

from hdbscan import HDBSCAN

#UMAP (Uniform Manifold Approximation and Projection)
#This is effective because it preserves the local structure of the data while reducing its dimensionality
from umap import UMAP

#gensim library for topic modeling and NLP tasks
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import random #Random library is used for generating random numbers and selecting random elements from a list
from gensim.utils import simple_preprocess #simple_preprocess is a function that tokenizes and preprocesses text data
from tqdm import tqdm #tqdm library is used for displaying progress bars in loops #Derived from the Arabic word "taqaddam" meaning progress

#specifying the device to use for PyTorch operations
#'cuda' is used for GPU acceleration otherwise 'cpu' is used
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.is_available()

def set_seed(seed: int = 42):
    random.seed(seed)              # Python
    np.random.seed(seed)           # NumPy
    torch.manual_seed(seed)        # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


#Specify the path to the file. Here, the file is the output of Lemmatization step
file = #input file for BERTopic - output of lemmatization 

df = pd.read_csv(file, encoding='utf-8-sig') #Reading the csv file keeping the encodings intact

#Specifying the exact column in the DataFrame that contains the lemmatized text data
#Converting the 'replaced_lemmas' column to a list for further processing
data = df['replaced_lemmas'].tolist()


#Remove Non-String Items because BERTopic requires a "list of strings" and non-string items can cause errors during processing.
#What are non-string items? Non-string items are elements in the list that are not of type string, such as integers, floats, or None.
#The final input to the BERTopic model is being called "data"
data = [str(item) for item in data if isinstance(item, str) or (isinstance(item, float) and not pd.isnull(item))]


# (a) Load the transformer model 

#jinaai/jina-embeddings-v3 is a pre-trained model for generating embeddings in Dutch language available on Hugging Face
#Based on our previous analysis, this model was the best performing model for out Dutch dataset
#Why do we need SetenceTransformer? Because it provides a simple interface for generating sentence embeddings using pre-trained models.

dutch_model = SentenceTransformer(
    "jinaai/jina-embeddings-v3", 
    trust_remote_code=True, # trust_remote_code=True allows the use of custom code from the model repository
    local_files_only = False # local_files_only=False allows downloading the model from the Hugging Face Hub if not available locally
   )

#(b) Generate Embeddings (output vectors)
#Now we pass the lemmatized text data to the model to generate embeddings
#What are embeddings? Embeddings are dense vector representations of text data that capture semantic meaning.
#The model will process the input data and generate embeddings for each text item in the list
embeddings = dutch_model.encode(
    data,
    show_progress_bar = True, #This displays a progress bar during the encoding process
    batch_size = 64 #Batch size is the number of samples processed before the model's internal parameters are updated.
)

np.save('jinaai_(Regex_Lemmatization)_entire data.npy', embeddings) #save the embeddings to a .npy file (a binary file format used by NumPy to store arrays efficiently
print("Embeddings saved successfully!") #Print that the embeddings have been saved successfully


#Dimensionality Reduction
#Initialize UMAP
#UMAP parameters: 
#n_neighbors: Controls the size of the local neighborhood used for manifold approximation
#n_components: The number of dimensions to reduce the data to (default is 2)
#metric: The distance metric used to compute distances in high-dimensional space
#random_state: Ensures reproducibility of results by setting a random seed
#min_dist: Controls how tightly UMAP packs points together in the low-dimensional space
#low_memory: If set to True, UMAP will use less memory at the cost of slower computation
umap_model = UMAP(n_neighbors=20, #Higher values to consider more neighbours and reduce outliers for large clusters
                  n_components=15, #The dimensionality of the embeddings after reducing them
                  metric = 'cosine', #the method used to compute the distances in high dimensional space
                  random_state = 42,
                  min_dist = 0.0, #reduce to allow tighter clusters 
                  )



#HDBSCAN parameters:
#min_cluster_size: The minimum number of data points required to form a cluster
#prediction_data: If True, allows predicting cluster labels for new data points later on
#metric: The distance metric used to compute distances in high-dimensional space
#cluster_selection_method: The method used to select clusters (e.g., 'eom' for excess of mass)
#Note: The metric should match the one used in UMAP if you change n_components in UMAP
#Initialize HDBSCAN model with the specified parameters

hdbscan_model = HDBSCAN(
    min_cluster_size = 5, #minimum number of data points to form a cluster
    prediction_data = True, #Needed to predict new points later on
    metric = 'euclidean', #Change if you increase the n_components in UMAP
    cluster_selection_method = 'eom'
    ) 


#Save the parameters used for UMAP and HDBSCAN

#Creating a Python dictionary "umap_parameters"
umap_parameters = {
    "n_neighbors" : umap_model.n_neighbors,
    "n_components" : umap_model.n_components,
    "metric" : umap_model.metric,
    "min_dist" : umap_model.min_dist,
    "random_state" : umap_model.random_state
}

with open("umap_parameters.txt", "w") as f:
    for key, value in umap_parameters.items():
        f.write(f"{key}: {value}\n")

print("UMAP parameters saved as a txt file.")

#Save HDBSCAN parameters as a txt file
hdbscan_parameters = {
        "min_cluster_size" : hdbscan_model.min_cluster_size,
        "prediction_data" : hdbscan_model.prediction_data,
        "metric" : hdbscan_model.metric,
        "cluster_selection_method" : hdbscan_model.cluster_selection_method
}    

with open("hdbscan_parameters.txt", "w") as f:
    for key, value in hdbscan_parameters.items():
        f.write(f"{key}: {value}\n")

print("hdbscan parameters saved succesfully!")

#Load Stop Words
stop_words = [word.strip("\n") for word in open(r"Stop list.txt", "r", encoding='utf-8-sig')]

#Initialize CountVectorizer with specified parameters
#CountVectorizer parameters:
# ngram_range: Specifies the range of n-grams to consider (1,1 means unigrams only)
# stop_words: Specifies the list of stop words to remove from the text data (already loaded)
# min_df: Minimum document frequency for a word to be included in the vocabulary (default is 1)
# max_df: Maximum document frequency for a word to be included in the vocabulary (default is 1.0)
vectorizer_model = CountVectorizer(
        ngram_range = (1,1),
        min_df = 2,
        stop_words = stop_words)


#Initialize ClassTfidfTransformer with specified parameters
#ClassTfidfTransformer parameters:
# bm25_weighting: If True, applies BM25 weighting to the term frequencies
        # BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query
        # BM25 weighting is a variant of TF-IDF that incorporates document length normalization and term saturation
# reduce_frequent_words: If True, reduces the impact of frequent words in the c-TFIDF representation
ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=True, 
        reduce_frequent_words=True
        )

#Create a representation model
# A representation model is used to generate topic representations based on the embeddings and c-TFIDF representation.
#It is used to create a mapping between the embeddings and the topics, allowing for better topic representation
#The representation model is used to generate topic representations based on the embeddings and c-TFIDF representation
#KeyBERTInspired is a representation model that generates topic representations based on the c-TFIDF representation
#It is inspired by the KeyBERT algorithm, which uses keyword extraction to create topic representations
##KeyBERTInspired is used to reduce the appearance of stop words and improve the topic representation
from bertopic.representation import KeyBERTInspired #Import KeyBERTInspired representation model
#Initialize the KeyBERTInspired representation model
representation_model = KeyBERTInspired() #To reduce the appearance of stop words and improve the topic representation



#Initialize BERTopic with the embedding model with all the previously defined parameters
#This includes the UMAP model, HDBSCAN model, CountVectorizer, ClassTfidfTransformer, and representation model
model = BERTopic(
    language="dutch", 
    verbose=True, 
    nr_topics = 150,       
    embedding_model = dutch_model, #Embedding model
    umap_model=umap_model, #Dimensionality reduction model
    hdbscan_model = hdbscan_model, # Clustering model    
    ctfidf_model = ctfidf_model, # ClassTfidfTransformer model
    vectorizer_model = vectorizer_model, # CountVectorizer model
    representation_model= representation_model, # Representation model
    calculate_probabilities=True #  allows BERTopic to calculate topic probabilities for each document
)


# Transforming the input data (a list of strings) into topics and probability
#Fit BERTopic on the Dutch data
topics, probs = model.fit_transform(data, embeddings) #fit_transform method returns both the topics and the probabilities
#Probabilities: a matrix of probabilities representing how likely a document belongs to each topic (to understand uncertainty)

#Save the model to a file
model.save('Model_jinaai_entire data', save_embedding_model = False)
print("Model saved successfully!")


# Save the BERTopic initialization parameters
bertopic_parameters = {
    "language": "dutch",
    "min_topic_size": 10,
    "verbose": True,
    "nr_topics": 150, 
    "umap_model": umap_model, 
    "hdbscan_model": hdbscan_model,  
    "embedding_model": dutch_model,  
    "vectorizer_model": vectorizer_model,
    "ctfidf_model": ctfidf_model, 
    "representation_model": representation_model,
     }

with open("BERTopic_parameters.txt", "w", encoding = "utf-8") as f:
    for key, value in bertopic_parameters.items():
        f.write(f"{key}: {value}\n")

print("BERTopic parameters saved successfully!")


#After fitting the model, we can access the topics and their counts of documents assigned to each topic
#"get.topic_freq()" returns the frequency of each topic in the model
#This includes the number of documents assigned to each topic during fitting
#What is the output of "get_topic_freq()"?
#The output is a pandas DataFrame with the following columns:
# Topic: the topic number (e.g., 0, 1, 2, etc.) and -1 refers to the outliers 
# Count: the number of documents assigned to each topic DURING FITTING
# First column: row identifiers in the DataFrame

#Topic count and the number of documents assigned to each topics before reduction of outliers
count_topic = model.get_topic_freq()
count_topic


#Create a copy of the original topics for reference
#Creating a copy of the original topics allows us to keep track of the original topic assignments
original_topics = topics.copy()

#Identifying the outlier topics (-1) in the original topics list allows us to reassign these documents to the most similar topic based on the Approximate Distribution Method
outlier_ids = [index for index, topic in enumerate(topics) if topic == -1]
#outlier_ids will be a list of indices where the topic is -1


#Reassign outlier documents to the most similar topic using the Approximate Distribution Method
#What is the Approximate Distribution Method?   
#This estimates the probability of each outlier belonging to any topic
topic_distr, _ = model.approximate_distribution(data)
#topic_distr is a matrix showing how likely each document is to belong to each topic

#What is the output of "approximate_distribution"?
#The output is a tuple containing two elements:
# 1. topic_distr: a numpy array of shape (n_documents, n_topics) representing the probability distribution of each document across topics
# 2. topic_labels: a list of topic labels corresponding to the topics in the topic_distr array

#Each row corresponds to a document, and each column corresponds to a topic
#The values in the array represent the probability of each document belonging to each topic
#The outlier topic (-1) is included in the topic_distr array, but its values will be very low compared to other topics


# Calculating the probability distribution for each outlier document (how likely it is to belong to each topic)

#Build final topics assignment
final_topics = [np.argmax(row) if row.max()>0 else -1 for row in topic_distr]

#Update BERTopic with the new topics
model.update_topics(data,          #The input data used to fit the model
                    topics=final_topics, #The updated topics list with reassigned outlier documents
                    vectorizer_model=vectorizer_model, # CountVectorizer model
                    ctfidf_model=ctfidf_model, # ClassTfidfTransformer model
                    representation_model=representation_model #KeyBERTInspired representation model
                    )

#Generate new topic labels based on the updated topics
new_labels = model.generate_topic_labels(
    nr_words=3, #Number of words to include in the topic label
    topic_prefix=True, #Whether to include a prefix for the topic label (e.g., "Topic 0")
    word_length=15,# Maximum length of each word in the topic label
    separator=" - " #Separator to use between words in the topic label
)

#Update the model with the new topic labels
model.set_topic_labels(new_labels)


#Print updated topic-term mappings #optional: (including weights)
print("Updated Topic Terms:")
for topic_id, term_list in model.get_topics().items():
    if topic_id == -1:
        continue
    sorted_terms = sorted(term_list, key=lambda x: x[1], reverse=True)
    words = [word for word, _ in sorted_terms]
    print(f"Topic {topic_id}: {', '.join(words)}")
    

#Print topic information 
topic_info = model.get_topic_info()
#topic_info is a pandas DataFrame containing information about each topic
# It includes the following columns:
# Topic: The topic ID (e.g., 0, 1, 2, etc.) and -1 refers to the outliers
# Size: The number of documents assigned to each topic
# Name: The name/label of each topic
#Top keywords for each topic

# Show representative documents per topic
print("\nRepresentative Documents:")
for topic_id in topic_info['Topic']:
    if topic_id == -1:
        continue
    # Use positional argument for number of documents to avoid TypeError
    docs = model.get_representative_docs(topic_id) #This shows representative documents per topic (the best ones not all the documents)
    print(f"\nTopic {topic_id} ({model.topic_labels_[topic_id]}):")
    for doc in docs:
        print(f" - {doc}")

#Save to CSV
topic_info.to_csv('topic_information.csv', index=False, encoding='utf-8-sig')

#Recompute the probabilities based on the updated topics
topics_final, probs_final = model.transform(data, embeddings= embeddings)

#Build DataFrame with numeric column names matching internal topic slots
probs_df = pd.DataFrame(probs_final)

#Get topic IDs from topic_info
topic_ids = topic_info["Topic"].tolist()


# pad zeros for dropped topics (the model still keeps slots)
if len(topic_ids) < probs_df.shape[1]:
    # fill extra columns as 'dropped'
    extra = [f"dropped_{i}" for i in range(probs_df.shape[1] - len(topic_ids))]
    col_labels = topic_ids + extra
else:
    col_labels = topic_ids[:probs_df.shape[1]]

probs_df.columns = col_labels

# drop outlier (-1) and any dropped columns
probs_df = probs_df[[c for c in probs_df.columns if c != -1 and not str(c).startswith("dropped_")]]
probs_df = probs_df.round(6)
probs_df.columns = [f"Topic_{c}" for c in probs_df.columns]

#build per-document table with assigned topic + top probability
doc_topics_df = pd.DataFrame({
    "Initial Topic": original_topics,
    "Final Topic": final_topics,
    "Max Probability": probs_df.max(axis=1)
}, index = df.index)

#combine the original document dataframe + probabilities
df_full = pd.concat([
    df, 
    doc_topics_df,
    probs_df
], axis=1)

#merge topic metadata
topic_info_final = model.get_topic_info().rename(columns={
    "Topic": "Final Topic",
    "Count": "Topic Size",
    "Name": "Topic Name"
})

df_full = df_full.merge(topic_info_final, on="Final Topic", how="left")

cols_to_drop = ["Topic Size", "Topic Name", "CustomName", "Representation", "Representative_Docs"]
df_full = df_full.drop(columns = cols_to_drop)

df_full.to_csv("Document_Topic_Distribution.csv", index=False, encoding="utf-8-sig")

#Word clouds
def freq_wordcloud(model, topic):
    word_freq = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(
        background_color="white",
        max_words = 100,
        width = 800,
        height = 400,
        random_state=42) #width and height are set to create a rectangular word cloud
    wc.generate_from_frequencies(word_freq)
    fig = plt.figure(figsize=(8,4), dpi =100)
    plt.imshow(wc,  interpolation = "bilinear")
    plt.axis("off")
    plt.title(f"Topic: {topic} ")
    return fig #Return the figure object to be saved later
    #plt.show()

#Generate and save WordCloud for all topics
with PdfPages('wordclouds_wordfreq.pdf') as pdf: #Opens a multi-page PDF file where all clouds will be saved
    for topic in sorted(set(final_topics)): #loops through each unique ID in final_topics
            #"set" ensures no duplicates; "sorted(...)" ensures consistent order
        if topic != -1: #skip the outlier topic
            fig = freq_wordcloud(model, topic) #creates a new, square figure (8x8 inches) for each word cloud
            pdf.savefig(fig) #saves the current wordcloud into the pdf
            plt.close(fig) #closes the figure after saving
    print("WordClouds saved to 'wordclouds.pdf' ") #Confirms the process is complete

#save the wordclouds as PNG files
output_dir = "wordfreq_wordclouds"
os.makedirs(output_dir, exist_ok=True)

for topic in sorted(set(final_topics)):
    if topic == -1:
        continue

    #print(f"Processing Topic: {topic}...")
    fig = freq_wordcloud(model, topic)
    output_path = os.path.join(output_dir, f"topic_{topic}_display.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

print("All word clouds saved as separate PNG files in the 'wordclouds/' folder.")
