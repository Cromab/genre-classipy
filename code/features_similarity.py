import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patheffects

sns.set()

X = pd.read_csv(r'../data/features_30_sec.csv')  # Replace with your actual data path

# Clean up the feature data
cols = """chroma_stft_mean,chroma_stft_var,rms_mean,rms_var,spectral_centroid_mean,spectral_centroid_var,spectral_bandwidth_mean,spectral_bandwidth_var,rolloff_mean,rolloff_var,zero_crossing_rate_mean,zero_crossing_rate_var,harmony_mean,harmony_var,perceptr_mean,perceptr_var,tempo""".split(',')
y = X['label']  # Genre labels
X = X[cols]  # Feature matrix

# Group by genre and calculate the mean feature vector for each genre
X['label'] = y
grouped = X.groupby('label').mean()

# Compute the Cosine Similarity matrix between the genre-specific mean vectors
cos_sim_matrix = cosine_similarity(grouped)

# Print the Cosine Similarity matrix
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cos_sim_matrix, annot=True, cmap="coolwarm_r", fmt=".3f", linewidths=0.5,
                 xticklabels=grouped.index, yticklabels=grouped.index,
                 annot_kws={"size": 13, "weight": "bold", "color": "white"})

# Add white text with black shadow effect
for text in ax.texts:
    # Increase font size if necessary
    text.set_fontsize(13)  # Change the size as needed
    text.set_path_effects([
        patheffects.withStroke(linewidth=2, foreground="black"),  # Black stroke for shadow effect
        patheffects.Normal()  # Normal text (white text)
    ])

# Set the title and labels
plt.title('Cosine Similarity Matrix of Genre Mean Feature Vectors')

# Save the plot as an image
plt.savefig(r'../../Images/similarity_features_10x10.png')

# Show the plot
plt.show()
