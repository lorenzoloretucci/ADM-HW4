#Import
from math import sqrt, floor
import numpy as np
import scipy.spatial.distance as metric

def random_centroids(ds, k): #Create random cluster centroids
    
    # Number of columns in dataset
    n = np.shape(ds)[1]

    # create a matrix of zeros for centroids initialization 
    centroids = np.mat(np.zeros((k,n)))

    # Create random centroids
    for j in range(n):
        min_j = min(ds[:,j])
        range_j = float(max(ds[:,j]) - min_j)
        centroids[:,j] = min_j + range_j * np.random.rand(k, 1)

    # Return centroids as array
    return centroids


def euc(A, B): # let's calculate de Euclidian Distance 

    return metric.euclidean(A,B)

def cluster(ds, k): #The principal k-means algorithm


    # Number of rows in dataset
    m = np.shape(ds)[0]

    # Hold the instance cluster assignments
    cluster_assignments = np.mat(np.zeros((m, 2)))

    # Initialize centroids
    cents = random_centroids(ds, k) # --> Use the Randon centroid function 

    # We need to preserve the original centroids
    cents_orig = cents.copy()

    changed = True
    iterations = 0 #let's start to calculate the number of iteration num_iter

    # Loop until no changes to cluster assignments
    while changed:

        changed = False

        # For every row in dataset
        for i in range(m):

            # Track minimum distance, and vector index of associated cluster
            min_dist = np.inf
            min_index = -1

            # Calculate distances
            for j in range(k):

                dist_ji = euc(cents[j,:], ds[i,:]) # --> Euclidian Distance
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j

            # Check if cluster assignment of instance has changed
            if cluster_assignments[i, 0] != min_index: 
                changed = True

            # Assign instance to appropriate cluster
            cluster_assignments[i, :] = min_index, min_dist**2

        # Update the new centroid location
        for cent in range(k):
            points = ds[np.nonzero(cluster_assignments[:,0].A==cent)[0]]
            cents[cent,:] = np.mean(points, axis=0)
        

        # Update the number of the iteretion 
        iterations += 1

    
    return cents, cluster_assignments, iterations, cents_orig

