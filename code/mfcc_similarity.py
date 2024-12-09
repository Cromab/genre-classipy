# %% Imports
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from matplotlib import patheffects
import seaborn as sns

sns.set()

# %% Read MFCCs for each genre
# Function to load and compute MFCC for each audio file
def compute_mfcc(file_path, n_mfcc=13, sr=22050):
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(file_path, sr=sr)
        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)  # Take the mean MFCCs over time
    except Exception as e:
        # Handle the error (e.g., corrupted file) and print the error message
        print(f"Error loading {file_path}: {e}")
        return None  # Return None if the file couldn't be processe

# Path to GTZAN dataset directory
gtzan_path = '../data/genres_original'

# List of genres in the GTZAN dataset
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Dictionary to hold MFCCs for each genre
genre_mfccs = {}

# Load the MFCCs for each genre
for genre in genres:
    genre_path = os.path.join(gtzan_path, genre)
    mfccs_list = []
    for filename in os.listdir(genre_path):
        file_path = os.path.join(genre_path, filename)
        if file_path.endswith('.wav'):
            mfcc = compute_mfcc(file_path)
            if mfcc is not None:  # Only add MFCC if the file was successfully processed
                mfccs_list.append(mfcc)
            else:
                print(f"Skipping corrupted file: {file_path}")
    genre_mfccs[genre] = np.array(mfccs_list) if mfccs_list else None

# %% Compute pairwise cosine similarity
# Compute pairwise cosine similarity between genres
genre_names = list(genre_mfccs.keys())
num_genres = len(genre_names)

# Initialize an empty matrix to store similarities
similarity_matrix = np.zeros((num_genres, num_genres))

# Calculate pairwise cosine similarities between the mean MFCCs of each genre
for i in range(num_genres):
    for j in range(num_genres):
        mfccs_i = genre_mfccs[genre_names[i]]
        mfccs_j = genre_mfccs[genre_names[j]]
        
        # Compute the mean MFCC for each genre
        mean_mfcc_i = np.mean(mfccs_i, axis=0)
        mean_mfcc_j = np.mean(mfccs_j, axis=0)
        
        # Compute cosine similarity
        similarity = 1 - cosine(mean_mfcc_i, mean_mfcc_j)  # Cosine similarity: 1 - distance
        similarity_matrix[i, j] = similarity


# %% Plot the similarity matrix
plt.figure(figsize=(10, 8))
ax = plt.gca()
cax = ax.imshow(similarity_matrix, cmap='coolwarm_r', interpolation='none')
plt.colorbar(cax)
plt.xticks(range(num_genres), genre_names, rotation=90)
plt.yticks(range(num_genres), genre_names)
plt.title('Cosine Similarity between Genres based on MFCCs')

# Add the similarity scores as text annotations on the heatmap
for i in range(num_genres):
    for j in range(num_genres):
        similarity = similarity_matrix[i, j]
        if not np.isnan(similarity):  # Only add text if the similarity is a number (not NaN)
            # Create text with a white font and black outline
            text = ax.text(j, i, f'{similarity:.2f}', ha='center', va='center', color='white', fontsize=14)
            
            # Add a black outline to the text
            text.set_path_effects([
                patheffects.withStroke(linewidth=3, foreground='black')  # Outline settings
            ])
plt.savefig(r'../../Images/similarity.png')
plt.show()
