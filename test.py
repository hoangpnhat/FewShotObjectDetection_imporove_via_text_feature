# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import torch
# def load_embeddings(file_path):
#     embeddings = {}
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             values = line.strip().split()
#             word = values[0]
#             vector = np.asarray(values[1:], dtype='float32')
#             embeddings[word] = vector
#     return embeddings
# def get_class_embed(class_names, model, include_bg=False):
#     with torch.no_grad():
#         semantic_features = []
#         for class_name in class_names:
#             semantic_features.append(np.loadtxt(f"datasets/{model}/{class_name}.txt"))
#         if include_bg:
#             semantic_features.append(np.loadtxt(f"datasets/{model}/background.txt"))
#         semantic_features = np.array(semantic_features)
            
#     return semantic_features

# def find_related_words(vector,embeddings, top_n=5):
#     # if word not in embeddings:
#     #     print(f"Word '{word}' not found in the embeddings.")
#     #     return []
    
#     # word_vector = embeddings[word]
#     word_vectors = np.array(list(embeddings.values()))
#     word_vector = vector
#     similarities = cosine_similarity(word_vector.reshape(1, -1), word_vectors)[0]
#     print(np.sort(similarities)[::-1][0:top_n])
#     related_indices = np.argsort(similarities)[::-1][0:top_n]

#     related_words = []
#     for index in related_indices:
#         related_word = list(embeddings.keys())[index]
#         # if related_word != word:
#         related_words.append(related_word)
#         if len(related_words) == top_n:
#             break
#     return related_words

# def calculate_average_vector(vector_list):
#     if len(vector_list) == 0:
#         return None

#     vector_dim = len(vector_list[0])
#     average_vector = np.zeros(vector_dim)

#     for vector in vector_list:
#         average_vector += vector

#     average_vector /= len(vector_list)
#     return average_vector
# def create_normalized_orthogonal_vector(vector):
#     # Generate a random vector of the same dimension as the given vector
#     random_vector = np.random.randn(*vector.shape)
#     # Subtract the projection of the given vector onto the random vector
#     orthogonal_vector = vector - np.dot(vector, random_vector) * random_vector
#     # Normalize the orthogonal vector
#     normalized_orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
#     return normalized_orthogonal_vector

# glove_file_path = '.word_vectors_cache/glove.6B.300d.txt'

# embeddings = load_embeddings(glove_file_path)
# name_class = ["aeroplane", "bicycle", "boat", "bottle", "car",
#         "cat", "chair", "diningtable", "dog", "horse",
#         "person", "pottedplant", "sheep", "train", "tvmonitor"
#     ]
# class_embed = get_class_embed(name_class,"glove")
# average_vector = np.array(calculate_average_vector(class_embed))
# neg_vector = np.negative(average_vector)
# normalized_orthogonal_vector = create_normalized_orthogonal_vector(average_vector)
# words = find_related_words(normalized_orthogonal_vector,embeddings)

# print(words)
import torch
from torch.nn import functional as F
gt = torch.load('gt_classes.pt')
# guide_gt = torch.load('Guided_gt_classes.pt')
guide_gt = torch.randint(16, (2048,), dtype=torch.int64)

# print(gt, gt.shape)
print(guide_gt,guide_gt.shape)

predict = torch.load('pred_logits.pt')
# print(gt.shape)
# print(predict.shape)
print(F.cross_entropy(predict,guide_gt,ignore_index=-1))