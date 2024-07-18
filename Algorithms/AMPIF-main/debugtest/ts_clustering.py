import numpy as np
from sklearn.cluster import KMeans



def get_Kmeans_clustering(num_cluster, fea):
    model = KMeans(n_clusters=num_cluster, random_state=0)
    cluster = model.fit_predict(fea)
    return cluster

def get_STquery_index(arr1, arr2):
    norm1 = np.linalg.norm(arr1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(arr2, axis=-1, keepdims=True)
    arr1_norm = arr1 / norm1
    arr2_norm = arr2 / norm2
    cos = np.dot(arr1_norm, arr2_norm.T)
    idx = np.argmax(cos, axis=0)
    return idx



def get_clustering(fea, num_cluster, support_rate, mode='factor'):

    B, L, E = fea.shape
    fea = np.nan_to_num(fea)
    fea = fea.reshape((B, L*E))
    support_len = int(B * support_rate)
    support_fea, query_fea = fea[: support_len], fea[support_len:]

    index_cluster = get_Kmeans_clustering(num_cluster, support_fea)

    num = 0
    bin_count = [0] * num_cluster
    cluster_mean = np.zeros([num_cluster, L * E])
    for i, c in enumerate(index_cluster):
        cluster_mean[c] += support_fea[i]
        bin_count[c] += 1
    for i in range(num_cluster):
        if bin_count[i] != 0:
            num += 1
            cluster_mean[i] /= bin_count[i]

    query_index = get_STquery_index(cluster_mean[:num], query_fea)
    index_cluster = np.concatenate((index_cluster, query_index), axis=0)
    return index_cluster

