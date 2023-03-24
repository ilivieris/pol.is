import leidenalg
import louvain
import numpy as np

def polis_leiden_onelayer(graph, optimizationfunction):
    '''
    Calculate clusters using Leiden algorithm for one-layer graph 
    
    Parameters:
      graph: the graph to detect communities on
      optimizationfunction: the optimization function for Leiden alg
    
    Returns:
      leidenClusters: the partition sequence detected by Leiden alg
    '''
    
    part = leidenalg.find_partition(
        graph, 
        optimizationfunction
    );

    leidenClusters = np.array(part.membership)

    return leidenClusters

# def polis_leiden_onelayer_iterative(graph, optimizationfunction, n_iterations):
#     # Calculate clusters using Leiden algorithm for one-layer graph, iteratively
#     #
#     # Parameters:
#     #   graph: the graph to detect communities on
#     #   optimizationfunction: the optimization function for Leiden alg
#     #   n_iterations (int): number of iterations
#     #
#     # Returns:
#     #   leidenClusters: the partition sequence detected by Leiden alg

#     part = optimizationfunction(graph)
#     diff = leidenalg.Optimiser().optimise_partition(part, n_iterations)

#     leidenClusters = np.array(part.membership)

#     return leidenClusters


def polis_louvain_onelayer(graph, weights, resolution_parameter = 1.5):
    '''
    Calculate clusters using Leiden algorithm for one-layer graph 
    
    Parameters:
      graph: the graph to detect communities on
      weights: weights on the graph
      optimizationfunction: the optimization function for Leiden alg
      resolution_parameter
    
    Returns:
      leidenClusters: the partition sequence detected by Leiden alg
    '''
    
    partition_type = louvain.RBConfigurationVertexPartition
    partition_kwargs = {"weights": weights, "resolution_parameter": resolution_parameter}
    part = louvain.find_partition(graph, partition_type, **partition_kwargs)
    Clusters = np.array(part.membership)

    return Clusters


def polis_leiden_twolayers(G_pos, G_neg, optimizationfunction):
    '''
    Calculate clusters using Leiden algorithm for two-layer graph 
    
    Parameters:
      G_pos: the positive (layer of the) graph
      G_neg: the negative (layer of the) graph
      optimizationfunction: the optimization function for Leiden alg
    
    Returns:
      leidenClusters: the partition sequence detected by Leiden alg
    '''
    optimiser = leidenalg.Optimiser()
    partition_pos = optimizationfunction(G_pos)
    partition_neg = optimizationfunction(G_neg)
    # Both layers are fed into Leiden. layer_weights=[1,-1] penalize negative layer
    # See also 4.1.1 in https://readthedocs.org/projects/leidenalg/downloads/pdf/latest/ 
    diff = optimiser.optimise_partition_multiplex(
                   partitions=[partition_pos, partition_neg],
                   layer_weights=[1,-1])
    
    leidenClusters = np.array(partition_pos.membership)

    return leidenClusters

# '''
# from collections import Counter
# G_pos, G_neg = generate_pos_neg_graphs(vals_all_in)
# lc = polis_leiden_twolayers(G_pos, G_neg, leidenalg.ModularityVertexPartition)
# lc_participants = lc[:vals_all_in.shape[0]]
# '''
# def polis_leiden_twolayers_iterative(G_pos, G_neg, optimizationfunction, n_iterations):
#     # Calculate clusters using Leiden algorithm for two-layer graph, iterative
#     #
#     # Parameters:
#     #   G_pos: the positive (layer of the) graph
#     #   G_neg: the negative (layer of the) graph
#     #   optimizationfunction: the optimization function for Leiden alg
#     #   n_iterations (int): number of iterations
#     #
#     # Returns:
#     #   leidenClusters: the partition sequence detected by Leiden alg

#     optimiser = leidenalg.Optimiser()
#     partition_pos = optimizationfunction(G_pos)
#     partition_neg = optimizationfunction(G_neg)
#     diff = optimiser.optimise_partition_multiplex(
#                    partitions=[partition_pos, partition_neg],
#                    layer_weights=[1,-1],
#                    n_iterations=n_iterations)
    
#     leidenClusters = np.array(partition_pos.membership)

#     return leidenClusters