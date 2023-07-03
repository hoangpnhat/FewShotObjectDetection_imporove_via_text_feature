import numpy as np
import torch

def get_class_embed(class_names, model, include_bg=False):
    with torch.no_grad():
        semantic_features = []
        for class_name in class_names:
            semantic_features.append(np.loadtxt(f"datasets/{model}/{class_name}.txt"))
        if include_bg:
            semantic_features.append(np.loadtxt(f"datasets/{model}/background.txt"))
        semantic_features = torch.tensor(np.array(semantic_features))
            
    return semantic_features.to('cuda').type(torch.float)

def create_normalized_orthogonal_tensor(tensor):
    # Generate a random tensor of the same shape as the given tensor
    random_tensor = torch.randn_like(tensor)

    # Subtract the projection of the given tensor onto the random tensor
    orthogonal_tensor = tensor - torch.dot(tensor.flatten(), random_tensor.flatten()) * random_tensor

    # Normalize the orthogonal tensor
    normalized_orthogonal_tensor = orthogonal_tensor / torch.norm(orthogonal_tensor)

    return normalized_orthogonal_tensor