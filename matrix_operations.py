import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

def create_adjacency_matrix(data, k, modality_type):
    modality_type="" # set to empty string for now
    matrix = np.zeros((len(data), len(data)))
    #maybe set different ks and weights depending on modality type

    match modality_type:
        case "location":
            # Nearest neighbors ok, but haversine distance would be better
            print(modality_type)
        case "time":
            # Absolute difference instead?
            print(modality_type)
        case "tags":
            # Categorical -> cosine similarity. but it's a list of tags, though
            print(modality_type)
        case _:
            nbrs = NearestNeighbors(n_neighbors=max(1,k), algorithm='auto').fit(data)
            distances, indices = nbrs.kneighbors(data)
            for i in range(len(data)):
                for j in indices[i]:  # i.e. for each of the k nearest neighbors
                    matrix[i, j] = 1
                    matrix[j, i] = 1
    return matrix

def fuse_matrices(matrices):
    fused_matrix = matrices[0].copy()
    for matrix in matrices[1:]:
        fused_matrix = np.logical_or(fused_matrix, matrix)
        fused_matrix = fused_matrix.astype(int)
    # print(f"Fused matrix shape: {fused_matrix.shape}")
    # print(f"Non-zero entries in fused matrix: {np.count_nonzero(fused_matrix)}")
    return fused_matrix

def perform_svd_reduction(matrix, reduced_dim, seed):
    # Apply SVD to reduce dimensionality
    svd_dim = min(reduced_dim, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=svd_dim)
    return svd.fit_transform(matrix)

def perform_clustering(matrix, n_clusters, seed):
    # Cluster with K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = kmeans.fit_predict(matrix)
    return clusters
