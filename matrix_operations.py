import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def create_adjacency_matrix(data, k, modality_type):
    # modality_type="" # set to empty string for now
    num_samples = len(data)
    matrix = np.zeros((num_samples, num_samples))
    #maybe set different ks and weights depending on modality type
    indices = []
    print(f"modality = {modality_type}, k = {k}, num_samples = {num_samples} ")

    match modality_type:
        case "location":
            # 'latitude', 'longitude'
            valid_indices = np.where((data[:, 0] != -1) & (data[:, 1] != -1))[0]
            valid_data = data[valid_indices]

            if len(valid_data) > 0:
                nbrs = NearestNeighbors(n_neighbors=min(k, len(valid_data)), metric=haversine_distance).fit(valid_data)
                indices = nbrs.kneighbors(valid_data, return_distance=False)

        case "time":
            # 'datetaken', 'dateupload'
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

        case "text":
            #'username','title','description','tags'

            # Get valid rows (ignore empty usernames, blank text)
            valid_indices = np.where(np.any(data != '', axis=1))[0]
            valid_data = data[valid_indices]
            num_valid = len(valid_data)

            username_sim = np.zeros((num_valid, num_valid))
            text_sim = np.zeros((num_valid, num_valid))
            tag_sim = np.zeros((num_valid, num_valid))

            # Username Similarity (Binary Match)
            for i in range(num_valid):
                for j in range(num_valid):
                    if i != j and valid_data[i, 0] == valid_data[j, 0] and valid_data[i, 0] != "":
                        username_sim[i, j] = 1
                        username_sim[j, i] = 1

            # Title + Description Similarity using TF-IDF + Cosine
            text_data = np.where(valid_data[:, 1] != '', valid_data[:, 1], ' ') + " " + np.where(valid_data[:, 2] != '', valid_data[:, 2], ' ')
            if np.any(text_data != " "):
                vectorizer = TfidfVectorizer()
                text_vectors = vectorizer.fit_transform(text_data)
                text_sim = cosine_similarity(text_vectors)
            
            # Tags Similarity using Jaccard Similarity
            tag_sets = [set(tags) if tags else set() for tags in valid_data[:, 3]]

            for i in range(num_valid):
                for j in range(i+1, num_valid):
                    tag_sim[i, j] = jaccard_similarity(tag_sets[i], tag_sets[j])
                    tag_sim[j, i] = tag_sim[i, j]

            scaler = MinMaxScaler()
            username_sim = scaler.fit_transform(username_sim)
            text_sim = scaler.fit_transform(text_sim)
            tag_sim = scaler.fit_transform(tag_sim)

            # Step 5: Combine All Scores (Weighted)
            combined_similarity = (
                (0.2 * username_sim) +
                (0.3 * text_sim) +
                (0.5 * tag_sim)
            )

            # Step 6: Find Nearest Neighbors
            indices = np.argsort(-combined_similarity, axis=1)[:, :k]

        case _:
            valid_indices = np.where(np.all(np.isfinite(data), axis=1))[0]  # Keep only valid rows
            valid_data = data[valid_indices]

            if len(valid_data) > 0:
                nbrs = NearestNeighbors(n_neighbors=max(1,k), algorithm='auto').fit(valid_data)
                indices = nbrs.kneighbors(valid_data, return_distance=False)
    
    print(f"len(valid_data) = {len(valid_data)}, len(indices) = {len(indices)}")
    
    if len(valid_data) > 0:
        print(f"first valid sample: {valid_data[0]}")

    # create adjacency matrix
    for i, row in enumerate(indices):
        for j in row:
            original_i = valid_indices[i]
            original_j = valid_indices[j]
            matrix[original_i, original_j] = 1
            matrix[original_j, original_i] = 1

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
