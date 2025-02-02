import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import hdbscan

def create_adjacency_matrix(data, modality_type):
    # modality_type="" # set to empty string for now
    num_samples = len(data)
    matrix = np.zeros((num_samples, num_samples))
    #maybe set different ks and weights depending on modality type
    indices = []
    
    match modality_type:
        case "location":
            # 'latitude', 'longitude'
            k=50
            valid_indices = np.where((data[:, 0] != -1) & (data[:, 1] != -1))[0]
            valid_data = data[valid_indices]

            if len(valid_data) > 0:
                nbrs = NearestNeighbors(n_neighbors=min(k, len(valid_data)), metric=haversine_distance).fit(valid_data)
                indices = nbrs.kneighbors(valid_data, return_distance=False)

        case "time":
            # 'datetaken', 'dateupload'
            k=150
            valid_indices = np.where(np.all(np.isfinite(data), axis=1))[0]
            valid_data = data[valid_indices]

            if len(valid_data) > 0:
                indices = []
                for i in range(len(valid_data)):
                    current_taken = valid_data[i, 0]  # datetaken
                    current_uploaded = valid_data[i, 1]  # dateupload

                    # Compute absolute differences for both timestamps
                    taken_diffs = np.abs(valid_data[:, 0] - current_taken)  # Difference in datetaken
                    upload_diffs = np.abs(valid_data[:, 1] - current_uploaded)  # Difference in dateupload

                    # Combine both differences (weights can be adjusted if necessary)
                    combined_diffs = taken_diffs + upload_diffs  # Simple sum; could be weighted

                    # Get k nearest in time based on combined metric
                    nearest_indices = np.argsort(combined_diffs)[:k]
                    indices.append(nearest_indices)

        case "username":
            indices = []

            # Get valid rows (ignore empty usernames)
            valid_indices = np.where(data[:, 0] != '')[0]
            valid_data = data[valid_indices, 0] # np.array(data[valid_indices], dtype=str).flatten()
            num_valid = len(valid_data)
            
            username_dict = {}

            for i in range(num_valid):
                username = valid_data[i]
                if username not in username_dict:
                    username_dict[username] = []
                username_dict[username].append(i)
            for i in range(num_valid):
                username = valid_data[i]
                indices.append(username_dict.get(username, []))

        case "tags":
            k = 50
            indices = []

            # Get valid rows (ignore empty tags)
            valid_indices = np.where(data[:, 0] != '')[0]
            valid_data = data[valid_indices, 0]
            num_valid = len(valid_data)

            # Convert tag lists into sets
            tag_sets = [set(tags) if tags else set() for tags in valid_data]

            # Compute Jaccard Similarity
            for i in range(num_valid):
                similarities = np.array([jaccard_similarity(tag_sets[i], tag_sets[j]) if i != j else -1 for j in range(num_valid)])
                indices.append(np.argsort(-similarities)[:k])

        case "text":
            #'title','description'
            k=50
            indices = []

            # Get valid rows (ignore empty usernames, blank text)
            valid_indices = np.where(np.any(data != '', axis=1))[0]
            valid_data = data[valid_indices]
            num_valid = len(valid_data)

            # Title + Description Similarity using TF-IDF + Cosine
            text_data = np.where(valid_data[:, 0] != '', valid_data[:, 0], ' ') + " " + np.where(valid_data[:, 1] != '', valid_data[:, 1], ' ')
            if np.any(text_data != " "):
                vectorizer = TfidfVectorizer()
                text_vectors = vectorizer.fit_transform(text_data)
                text_sim = cosine_similarity(text_vectors)

                indices = np.argsort(-text_sim, axis=1)[:, :k]
            else:
                indices = [[] for _ in range(num_valid)]

        case _:
            k=50
            valid_indices = np.where(np.all(np.isfinite(data), axis=1))[0]  # Keep only valid rows
            valid_data = data[valid_indices]

            if len(valid_data) > 0:
                nbrs = NearestNeighbors(n_neighbors=max(1,k), algorithm='auto').fit(valid_data)
                indices = nbrs.kneighbors(valid_data, return_distance=False)
    
    print(f"modality = {modality_type}, num_samples = {num_samples}, num_valid = {len(valid_data)}")

    # if len(valid_data) > 0:
    #     print(f"first valid sample: {valid_data[0]}")

    # create adjacency matrix
    for i, row in enumerate(indices):
        for j in row:
            original_i = valid_indices[i]
            original_j = valid_indices[j]
            if original_i != original_j:
                matrix[original_i, original_j] = 1
                # matrix[original_j, original_i] = 1

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
    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    return svd.fit_transform(matrix)

def perform_clustering(matrix, n_clusters, seed):
    # Cluster with K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = kmeans.fit_predict(matrix)
    return clusters

# def perform_dbscan_clustering(data, eps=0.5, min_samples=5):
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
#     clusters = dbscan.fit_predict(data)
#     return clusters

def perform_hdbscan_clustering(data, min_cluster_size=5, min_samples=2):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    clusters = clusterer.fit_predict(data)
    return clusters

def jaccard_similarity(set1, set2):
    if not set1 or not set2:  # If either set is empty, similarity is 0
        return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def haversine_distance(location1, location2):
    lat1, lon1  = location1
    lat2, lon2 = location2

    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r

def incremental_dbscan_clustering(data, previous_centroids, previous_labels, eps=0.5, min_samples=5):
    # Convert input to 2D NumPy array if needed
    if not isinstance(data, np.ndarray):
        print(f"WARNING: Converting `data` to NumPy array, original type: {type(data)}")
        data = np.array(data, dtype=np.float32)

    if len(data.shape) != 2:
        print(f"ERROR: DBSCAN expects 2D data but got shape {data.shape}")
        return None, previous_centroids, previous_labels

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(data)
    
    unique_clusters = set(labels) - {-1}  # Exclude noise (-1)
    new_centroids = np.array([data[labels == cluster].mean(axis=0) for cluster in unique_clusters])
    
    # Match new clusters to previous clusters
    if previous_centroids is not None and len(previous_centroids) > 0:
        print(f"previous_labels: {previous_labels}, size={len(previous_labels)}")

        # Compute distances between old and new cluster centroids
        distances = cdist(new_centroids, previous_centroids)
        
        # Find closest old cluster for each new cluster
        matches = np.argmin(distances, axis=1)  
        print(f"matches: {matches}")
        
        # Remap labels to maintain consistency
        new_label_mapping = {new: previous_labels[old] if old < len(previous_labels) else -1 for new, old in enumerate(matches)}
        labels = np.array([new_label_mapping[label] if label in new_label_mapping else label for label in labels])
    
    new_labels = np.unique(labels)
    
    return labels, new_centroids, new_labels
