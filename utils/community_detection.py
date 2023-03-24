import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import kneighbors_graph

from utils.utils import sparsity_aware_dist
from utils.utils import polis_pca, polis_umap

import leidenalg
from collections import Counter
from utils.graph_generation import generate_kneighbors_graph, generate_pos_neg_graphs, generate_dual_graph
from utils.graph_clustering import polis_leiden_onelayer, polis_leiden_twolayers


def show_embedding(space, color, title):
    '''    
    Helper function to plot community data as color on an embedding space
    
    Parameters:
      space: the embedding space to plot data on (e.g. UMAP, PCA)
      color: the community data to plot as color in the embedding
      title: the plot's title
    '''

    fig, ax = plt.subplots(figsize=(7,5))
    plt.sca(ax)
    # NB: palette contains only 20 colors ('tab20')
    # (User needs to artificially limit communities to <20)
    ax.scatter(
        x=space[:,0],
        y=space[:,1],
        c=color,
        cmap="tab20",
        s=5
    )
    ax.set_title(title, fontsize=14)
    plt.show()




def polis_subconversation_kNN_graph(dataframe=None, neighbors=8):
    '''
    Experimentation with and plotting of different community detection 
    scenarios, on variations of the same dataset. 
    (Approximate stand-in for original polis_subconversation function)
    
    Parameters
    ----------
      dataframe: a participant-votes dataframe
      neighbors: number of neighbors

    Returns
    -------
      leidenClust: Clusters obtaining the communities
    '''
    
    print('[INFO] Application of PCA', end=': ')
    coords, explained_variance = polis_pca(dataframe, 2)
    print('Complete')
    print("Explained variance:", explained_variance)

    print('[INFO] Application of UMAP', end=': ')
    embedding = polis_umap(dataframe, 10);
    print('Complete')


    
    # K-neighbors, one layer, modularity
    # -----------------------------------------------------------------------------------------------        
    print(f'[INFO] Community detection (K-neighbors, one layer, modularity)', end=': ')
    G_kneighbors, weights = generate_kneighbors_graph(dataframe=dataframe, neighbors=neighbors)
    leidenClusters = polis_leiden_onelayer(G_kneighbors, leidenalg.ModularityVertexPartition)
    print('Complete')    
    # Print the sequence and nr of communities detected
    print("[INFO] Number of communities detected: {}".format(len(Counter(leidenClusters).keys())))
    print("[INFO] Sequence of communities detected: {}".format(Counter(np.array(leidenClusters))))
     # Show Embedding    
    show_embedding(coords, leidenClusters, "Leiden, PCA, kNN graph, one layer, modularity")
    show_embedding(embedding, leidenClusters, "Leiden, UMAP, kNN graph, one layer, modularity")


    # # K-neighbors, one layer, CPM
    # # -----------------------------------------------------------------------------------------------        
    # print(f'[INFO] Community detection (K-neighbors, one layer, CPM)', end=': ')
    # G_kneighbors, weights = generate_kneighbors_graph(dataframe=dataframe, neighbors=neighbors)
    # leidenClusters = polis_leiden_onelayer(G_kneighbors, leidenalg.CPMVertexPartition)
    # print('Complete')    
    # # Print the sequence and nr of communities detected
    # print("[INFO] Number of communities detected: {}".format(len(Counter(leidenClusters).keys())))
    # print("[INFO] Sequence of communities detected: {}".format(Counter(np.array(leidenClusters))))
    # # Show Embedding    
    # show_embedding(coords, leidenClusters, "Leiden, PCA, kNN graph, one layer, CPM")
    # show_embedding(embedding, leidenClusters, "Leiden, UMAP, kNN graph, one layer, CPM")

       

    # # Show clustermap
    # dataframe['cluster_assignments'] = leidenClusters
    # clusters_by_comments_means = dataframe.groupby('cluster_assignments').agg('mean')
    # sns.heatmap(clusters_by_comments_means, cmap="RdYlBu")
    # sns.clustermap(clusters_by_comments_means, cmap="RdYlBu")

    return leidenClusters, coords, embedding






def polis_subconversation_Adjacency_matrix(dataframe=None, neighbors=8):
    '''
    Experimentation with and plotting of different community detection 
    scenarios, on variations of the same dataset. 
    (Approximate stand-in for original polis_subconversation function)
    
    Parameters
    ----------
      dataframe: a participant-votes dataframe
      neighbors: number of neighbors

    Returns
    -------
      leidenClust: Clusters obtaining the communities
    '''
    
    print('[INFO] Application of PCA', end=': ')
    coords, explained_variance = polis_pca(dataframe, 2)
    print('Complete')
    print("Explained variance:", explained_variance)

    print('[INFO] Application of UMAP', end=': ')
    embedding = polis_umap(dataframe, 10);
    print('Complete')



    # Adjacency matrix, one layer, modularity
    # -----------------------------------------------------------------------------------------------
    print(f'[INFO] Community detection (Adjacency matrix, one layer, modularity)', end=': ')
    G = generate_dual_graph(dataframe)
    leidenClusters = polis_leiden_onelayer(G, leidenalg.ModularityVertexPartition)
    # Adjustment in case of > 20 communities (i.e. limited color palette)
    lc_show = []
    for x in leidenClusters[:embedding.shape[0]]:
      lc_show.append(x) if x < 20 else lc_show.append(20)
    # Print the sequence and nr of communities detected
    print("Nr of communities detected: {}".format(len(Counter(lc_show).keys())))
    print("Seq of communities detected: {}".format(Counter(np.array(lc_show))))
    show_embedding(embedding, lc_show, "Leiden, UMAP, adjacency matrix, one layer, modularity")
    show_embedding(coords, lc_show, "Leiden, PCA, adjacency matrix, one layer, modularity")
     


    # # Adjacency matrix, one layer, CPM
    # # -----------------------------------------------------------------------------------------------
    # print(f'[INFO] Community detection (Adjacency matrix, one layer, CPM)', end=': ')
    # G = generate_dual_graph(dataframe)
    # leidenClusters = polis_leiden_onelayer(G, leidenalg.CPMVertexPartition)
    # # Adjustment in case of > 20 communities (i.e. limited color palette)
    # lc_show = []
    # for x in leidenClusters[:embedding.shape[0]]:
    #   lc_show.append(x) if x < 20 else lc_show.append(20)
    # # Print the sequence and nr of communities detected
    # print("Nr of communities detected: {}".format(len(Counter(lc_show).keys())))
    # print("Seq of communities detected: {}".format(Counter(np.array(lc_show))))
    # show_embedding(embedding, lc_show, "Leiden, UMAP, adjacency matrix, one layer, CPM")
    # show_embedding(coords, lc_show, "Leiden, PCA, adjacency matrix, one layer, CPM")
    
    # # Show clustermap
    # dataframe['cluster_assignments'] = leidenClusters
    # clusters_by_comments_means = dataframe.groupby('cluster_assignments').agg('mean')
    # sns.heatmap(clusters_by_comments_means, cmap="RdYlBu")
    # sns.clustermap(clusters_by_comments_means, cmap="RdYlBu")

    return leidenClusters, coords, embedding






def polis_subconversation_Adjacency_matrix_two_layers(dataframe=None, neighbors=8):
    '''
    Experimentation with and plotting of different community detection 
    scenarios, on variations of the same dataset. 
    (Approximate stand-in for original polis_subconversation function)
    
    Parameters
    ----------
      dataframe: a participant-votes dataframe
      neighbors: number of neighbors

    Returns
    -------
      leidenClust: Clusters obtaining the communities
    '''
    
    print('[INFO] Application of PCA', end=': ')
    coords, explained_variance = polis_pca(dataframe, 2)
    print('Complete')
    print("Explained variance:", explained_variance)

    print('[INFO] Application of UMAP', end=': ')
    embedding = polis_umap(dataframe, 10);
    print('Complete')



    # Adjacency matrix, two layers, modularity
    # -----------------------------------------------------------------------------------------------
    print(f'[INFO] Community detection (Adjacency matrix, two layers, modularity)', end=': ')
    G_pos, G_neg = generate_pos_neg_graphs(dataframe)
    leidenClusters = polis_leiden_twolayers(G_pos, G_neg, leidenalg.ModularityVertexPartition)
    # Adjustment in case of > 20 communities (i.e. limited color palette)
    lc_show = []
    for x in leidenClusters[:embedding.shape[0]]:
      lc_show.append(x) if x < 20 else lc_show.append(20)
    # Print the sequence and nr of communities detected
    print("Nr of communities detected: {}".format(len(Counter(lc_show).keys())))
    print("Seq of communities detected: {}".format(Counter(np.array(lc_show))))
    show_embedding(embedding, lc_show, "Leiden, UMAP, adjacency matrix, two layers, modularity")
    show_embedding(coords, lc_show, "Leiden, PCA, adjacency matrix, two layers, modularity")
     


    # # Adjacency matrix, two layers, modularity
    # # -----------------------------------------------------------------------------------------------
    # print(f'[INFO] Community detection (Adjacency matrix, two layers, CPM)', end=': ')
    # G_pos, G_neg = generate_pos_neg_graphs(dataframe)
    # leidenClusters = polis_leiden_twolayers(G_pos, G_neg, leidenalg.CPMVertexPartition)
    # # Adjustment in case of > 20 communities (i.e. limited color palette)
    # lc_show = []
    # for x in leidenClusters[:embedding.shape[0]]:
    #   lc_show.append(x) if x < 20 else lc_show.append(20)
    # # Print the sequence and nr of communities detected
    # print("Nr of communities detected: {}".format(len(Counter(lc_show).keys())))
    # print("Seq of communities detected: {}".format(Counter(np.array(lc_show))))
    # show_embedding(embedding, lc_show, "Leiden, UMAP, adjacency matrix, two layers, CPM")
    # show_embedding(coords, lc_show, "Leiden, PCA, adjacency matrix, two layers, CPM")



    # # Show clustermap
    # dataframe['cluster_assignments'] = leidenClusters
    # clusters_by_comments_means = dataframe.groupby('cluster_assignments').agg('mean')
    # sns.heatmap(clusters_by_comments_means, cmap="RdYlBu")
    # sns.clustermap(clusters_by_comments_means, cmap="RdYlBu")

    return leidenClusters, coords, embedding