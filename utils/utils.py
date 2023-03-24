import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def count_finite(row):
    '''
        Count the number of finite values

        Parameters
        ----------
        row: row from a DataFrame

        Returns
        -------
        The number of finite values
    '''
    finite = np.isfinite(row) # boolean array of whether each entry is finite
    return sum(finite) # count number of True values in `finite`



def select_rows(df=None, val_fields=None, threshold=7):
    '''
        Remove participants with less than N (threshold) votes.

        Parameters
        ----------
        df: DataFrame with participants votes
        val_fields: variable fields
        threshold: threshold

        Returns
        -------
        DataFrame after removing the participants with less than N (threshold) votes
    '''
    
    number_of_votes = df[val_fields].apply(count_finite, axis=1)
    valid = number_of_votes >= threshold
    
    return df[valid]


def polis_pca(dataframe, components):
    '''
        Apply PCA on opinion matrix

        Parameters
        ----------
        dataframe: DataFrame containing opinion matrix
        components: Number of components for PCA

        Returns
        -------
        coords: Principal axes in feature space, representing the directions of maximum variance in the data
        explained_variance: The amount of variance explained by each of the selected components
    '''
    pca_object = PCA(n_components=components) ## pca is apparently different, it wants 
    pca_object = pca_object.fit(dataframe.T) ## .T transposes the matrix (flips it)
    coords = pca_object.components_.T ## isolate the coordinates and flip 
    explained_variance = pca_object.explained_variance_ratio_

    return coords, explained_variance



# @numba.njit()
def sparsity_aware_dist(a, b):
    n_both_seen = len(a) - (np.isnan(a) | np.isnan(b)).sum()
    return (n_both_seen - (a == b).sum() + 1) / (n_both_seen + 2)



def polis_umap(dataframe, neighbors, metric='euclidean'):
    '''
        Apply UMAP on opinion matrix

        Parameters
        ----------
        dataframe: DataFrame containing opinion matrix
        neighbors: Number of neighbors for UMAP
        metric: distance metric (Default: sparsity_aware_dist)

        Returns
        -------
        embeddings
    '''
    
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        metric=metric,        
        init='random',
        min_dist=0.1,
        spread=1.0,
        local_connectivity=3.0,
    )
    
    return reducer.fit_transform(dataframe.values)



def comment_visualization(dataframe, comment_id, comment, coords, embedding):
    '''
        Plot coords/embedding based on the votes for a specific comment

        Parameters
        ----------
        dataframe: DataFrame with participants votes 
        comment_id: comment id
        comment: comment (text)
        coords: 2D-PCA coordinates
        embedding: embedding from UMAP
    '''    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    colorMap = {-1:'#A50026', 1:'#313695', 0:'#FEFEC050'}

    ax[0].scatter(
        x=coords[:,0],
        y=coords[:,1],
        c=dataframe[str(comment_id)].apply(lambda x: colorMap[x]),
        s=10
    )

    ax[1].scatter(
        x=embedding[:,0],
        y=embedding[:,1],
        c=dataframe[str(comment_id)].apply(lambda x: colorMap[x]),
        s=10
    )   

    print("Comment No " + str(comment_id) + ": " + comment)
    ax[0].set_title('PCA', fontsize=14)
    ax[1].set_title('UMAP', fontsize=14)
    plt.show()