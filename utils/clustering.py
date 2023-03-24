import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def polis_kmeans_(dataframe, n_clusters=2):
    '''
        Given a dataframe, returns the found lables/clusters and the corresponding centers

        Parameters
        ----------
        dataframe: DataFrame with participants and their votes
        n_clusters: number of clusters for k-Means

        Returns
        -------
        labels of the clusters 
        center of the clusters
    '''
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dataframe)
    
    return kmeans.labels_, kmeans.cluster_centers_


def plot_embedding_with_clusters(embedding_, labels_):
    '''
        Plot embedding with clusters
    '''
    print("Plotting PCA embeddings with K-means, K="+str(np.max(labels_)+1))
    fig, ax = plt.subplots(figsize=(7,5))
    plt.sca(ax)
    ax.scatter(
        x=embedding_[:,0],
        y=embedding_[:,1],
        c=labels_,
        cmap="tab20",
        s=5
    )    