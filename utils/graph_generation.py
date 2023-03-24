import numpy as np
import pandas as pd
import igraph as ig
from sklearn.neighbors import kneighbors_graph
from utils.utils import sparsity_aware_dist

def generate_kneighbors_graph(dataframe, neighbors, metric=sparsity_aware_dist):
    '''
        Generate k-neighbors unipartite graphs using dataframe data
        
        Parameters
        ----------
        dataframe: a participant-votes dataframe
        neighbors (int): number of neighbors (k)
        metric: distance metric
        
        Returns
        -------
        G: participant-participant unipartite graph of k-neighbors connectivity
        weights: weights
    '''    

    A = kneighbors_graph(
        dataframe.values, 
        neighbors, 
        mode="connectivity", 
        metric=sparsity_aware_dist, 
        p=3, 
        metric_params=None, 
        include_self=True, 
        n_jobs=None
    )
    
    print("Dataframe shape: {}".format(dataframe.values.shape))
    print("Kneighbor graph shape: {}".format(A.shape))

    sources, targets = A.nonzero()
    weights = A[sources, targets]
    if isinstance(weights, np.matrix): # ravel data
            weights = weights.A1

    g = ig.Graph(directed=False)
    g.add_vertices(A.shape[0])  # each observation is a node
    edges = list(zip(sources, targets))
    g.add_edges(edges)
    g.es['weight'] = weights
    weights = np.array(g.es["weight"]).astype(np.float64)

    return g, weights


def generate_pos_neg_graphs(dataframe):
    '''
        Generate Pos/Neg graphs

        Parameters
        ----------
        dataframe: DataFrame with participants votes

        Returns
        -------
        Pos/Neg graphs
    '''
    # Generate two-layer multiplex bipartite graphs using dataframe data
    #
    # Parameters:
    #   dataframe: a participant-votes dataframe
    #
    # Returns:
    #   G_pos: participant-votes bipartite graph of positive votes (i.e. positive layer)
    #   G_neg: participant-votes bipartite graph of negative votes (i.e. negative layer)
    A = dataframe.values
    
    # Convert 'bipartite' matrix A (rows=participants, columns=votes) 
    # into unipartite form, by adding votes rows and participant columns.
    #
    # The new rectangular matrix A_sym can be written in block form as:
    # A_sym = [ 0     A ]
    #         [ A.T   0 ]
    A_sym = np.block([[np.zeros((A.shape[0], A.shape[0])), A], [np.transpose(A), np.zeros((A.shape[1], A.shape[1]))] ])

    # G_pos includes only positive edges (positive weights in A_sym)
    G_pos = ig.Graph.Adjacency((A_sym > 0).tolist(), mode="undirected")
    # G_neg includes only negative edges (negative weights in A_sym)
    G_neg = ig.Graph.Adjacency((A_sym < 0).tolist(), mode="undirected")

    # Partition of nodes in two classes (bipartite graph) - optional info
    # Class 0: participants - first A.shape[0] nodes
    # Class 1: votes - subsequent A.shape[1] nodes
    G_pos.vs['type'] = np.block([np.ones(A.shape[0]), np.zeros(A.shape[1])])
    G_neg.vs['type'] = np.block([np.ones(A.shape[0]), np.zeros(A.shape[1])])

    return G_pos, G_neg

def generate_dual_graph(dataframe):
    '''
        Generate dual graph

        Parameters
        ----------
        dataframe: DataFrame with participants votes

        Returns
        -------
        Dual graph
    '''    
    # Generate one-layer bipartite graph using dataframe data
    #
    # Parameters:
    #   dataframe: a participant-votes dataframe
    #
    # Returns:
    #   G: participant-votes bipartite graph of pos/neg votes (i.e. one layer)

    A = dataframe.values
    
    A_sym = np.block([[np.zeros((A.shape[0], A.shape[0])), A], [np.transpose(A), np.zeros((A.shape[1], A.shape[1]))] ])
    g = ig.Graph.Adjacency((A_sym != 0).tolist(), mode="undirected")
    g.es['weight'] = A_sym[A_sym.nonzero()] # Possibly not needed

    # Partition of nodes in two classes (bipartite graph) - optional info
    # Class 0: participants - first A.shape[0] nodes
    # Class 1: votes - subsequent A.shape[1] nodes
    g.vs['type'] = np.block([np.ones(A.shape[0]), np.zeros(A.shape[1])])

    # Naming nodes for easy retrieval
    for idx, v in enumerate(g.vs):
      v['name'] = idx

    return g


