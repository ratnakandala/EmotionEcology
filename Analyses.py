# import modules
import pandas as pd
import numpy as np

# import data (topic distribution matrix output including word/lemma counts from previous step)
data = pd.read_csv('Topic Distribution Matrix_counts_merged.csv', encoding='utf-8-sig')

# initialize vector
unique_clusters_by_word = {}

# loop through and get number of unique clusters labeled by each emotion word
for col in data.columns[141:]: # specify column where emotion word counts begin
    subset = data[data[col] >= 1]
    unique_clusters = subset['Final Topic'].nunique()
    unique_clusters_by_word[col] = unique_clusters

# output results
output = pd.DataFrame(list(unique_clusters_by_word.items()), columns=['Emotion_Word', 'Unique_Topics'])
output.to_csv('unique_counts.csv', index=False)

# loop through and get number of unique emotion words associated with each cluster
emotion_words_by_cluster = {}
unique_clusters = data['Final Topic'].unique()
for cluster in unique_clusters:
    subset = data[data['Final Topic'] == cluster]
    emotion_words = subset.iloc[:, 141:] # specify column where emotion word counts begin
    emotion_words = emotion_words.sum(axis=0)
    emotion_words = emotion_words[emotion_words >= 1]
    emotion_words_by_cluster[cluster] = emotion_words.size
output = pd.DataFrame(list(emotion_words_by_cluster.items()), columns=['Topic', 'Unique_Words'])
output.to_csv('unique_words.csv', index=False)

# define Gini function
def gini_coefficient(values):
    values = np.array(values)
    values_sorted = np.sort(values)
    n = len(values)
    cumulative_values_sum = np.cumsum(values_sorted, dtype=np.float64)
    cumulative_index = np.arange(1, n + 1)
    gini_numerator = np.sum((2 * cumulative_index - n - 1) * values_sorted)
    gini_denominator = n * np.sum(values_sorted)

    return 1-(gini_numerator / gini_denominator) if gini_denominator != 0 else 0

# initialize vector
gini_results = {}

# Step 1: Binarize the data (any value > 0 is set to 1)
data_binarized = (data.iloc[:, 22:140] > 0.01).astype(int) # specify columns containing topic loading probabilities
data_binarized['connectionId'] = data['connectionId']

# Step 2: Sum across rows for each participant
summed_data = data_binarized.groupby('connectionId').sum()

# Step 3: Calculate Gini coefficient for each participant
for PPID, values in summed_data.iterrows():
    gini_value = gini_coefficient(values)
    gini_results[PPID] = gini_value

# Convert results to a DataFrame for saving
gini = pd.DataFrame(list(gini_results.items()), columns=['connectionId', 'Gini_Coefficient'])

# Save to a CSV file
gini.to_csv('gini_results.csv', index=False)
summed_data.to_csv('summed.csv', index=False)

