import pickle
import sys
import numpy as np
import torch 
from torch import nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

classes = ["aeroplane", "bicycle", "boat", "bottle", "car",
        "cat", "chair", "diningtable", "dog", "horse",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
        "bird", "bus", "cow", "motorbike", "sofa"
    ]
def get_class_embed(class_names, model, include_bg=False):
    with torch.no_grad():
        semantic_features = []
        for class_name in class_names:
            semantic_features.append(np.loadtxt(f"datasets/{model}/{class_name}.txt"))
        if include_bg:
            semantic_features.append(np.loadtxt(f"datasets/{model}/background.txt"))
        # semantic_features = torch.tensor(np.array(semantic_features))
            
    return semantic_features
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

# Generate 16 random tensors as an example
num_tensors = len(classes)

tensors = get_class_embed(classes,'glove')


# background = tensors[-1]
# # Calculate cosine similarity with the other 15 tensors
# cosine_similarities = [round(cosine_similarity([background], [tensor])[0][0],2) for tensor in tensors[:-1]]
# # Plot the cosine similarities
# plt.figure(figsize=(8, 5))
# sns.barplot(cosine_similarities)
# plt.xlim(0.65, 1)

# plt.title('Cosine Similarity of "Background" vector with each classes vector',fontsize=14)
# plt.yticks(np.arange(num_tensors), classes)
# plt.xlabel('Cosine Similarity',fontsize=12)



# Calculate cosine similarity matrix
cosine_sim_matrix = np.zeros((num_tensors, num_tensors))
for i in range(num_tensors):
    for j in range(num_tensors):
        cosine_sim_matrix[i, j] = round(cosine_similarity([tensors[i]], [tensors[j]])[0, 0],2)
# import pdb; pdb.set_trace()

# Convert cosine similarity values to distances
cosine_distance_matrix = 1 - cosine_sim_matrix
# import pdb; pdb.set_trace()
#take a lower triangle matrix
matrix_tril = np.triu(cosine_sim_matrix)

# Display confusion matrix
plt.figure(figsize=(12, 10))
# sns.heatmap(cosine_sim_matrix, cmap='viridis', mask=mask, annot=True, fmt=".2f", )
sns.heatmap(cosine_sim_matrix,mask=matrix_tril, cmap=sns.diverging_palette(220, 20, as_cmap=True)
,cbar_kws={"shrink": 0.8},annot=True)
plt.title('Cosine Similarity Matrix')

plt.xticks(np.arange(num_tensors)+0.5, classes, rotation=90,horizontalalignment='center')    
plt.yticks(np.arange(num_tensors)+0.5, classes, rotation=0)

plt.savefig('Similarity_matrix_GLOVE.png')

