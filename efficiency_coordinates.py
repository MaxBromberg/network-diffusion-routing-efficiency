import numpy as np
from networkx import to_directed, from_numpy_matrix, to_numpy_array, shortest_path, DiGraph, grid_graph, fast_gnp_random_graph
from scipy.sparse.linalg import inv
from scipy.sparse import diags, eye, csc_matrix
from utils import matrix_normalize


# Effective Distance Evaluation: --------------------------------------------------------------------------------
def Random_Walker_Effective_Distance(A, source=None, target=None, parameter=1, via_numpy=False, return_errors=False):
    """
    Directly adapted/taken from Dr. Koher's version _[2], computes the random walk effective distance _[1]

    Parameters
    ----------
        A: 2d ndarray of int/float
            Adjacency matrix

        source : int or None
           If source is None, the distances from all nodes to the target is calculated
           Otherwise the integer has to correspond to a node index

        target : int or None
            If target is None, the distances from the source to all other nodes is calculated
            Otherwise the integer has to correspond to a node index

        parameter : float
            compound delta which includes the infection and recovery rate alpha and beta, respectively,
            the mobility rate kappa and the Euler-Mascheroni constant lambda:
                log[ (alpha-beta)/kappa - lambda ]

        via_numpy : bool, default: False
            if True, uses numpy linear algebra operations. Potentially more efficient for small, dense networks

        return_errors: bool, default: False
            if True, returns the number of singular fundamental matrix errors
             found through RWED calculation as second element of return tuple.


    Returns
    --------
        random_walk_distance : ndarray or float
            If source and target are specified, a float value is returned that specifies the distance.

            If either source or target is None a numpy array is returned.
            The position corresponds to the node ID.
            shape = (Nnodes,)

            If both are None a numpy array is returned.
            Each row corresponds to the node ID.
            shape = (Nnodes,Nnodes)

        _singular_fundamental_matrix_errors : int, optional
            Counts the number of times there are 0s present the in fundamental matrix Z arising from the original A.
            If instances of 0s occur in Z, they are in practice replaced by 1e-100 ~ 0

    Notes
    -----
     [1] The hidden geometry of complex, network-driven contagion phenomena.
     Brockmann, Dirk, and Dirk Helbing
     science 342, no. 6164 (2013): 1337-1342.

     [2] https://github.com/andreaskoher/effective_distance
    """
    if np.all(np.isclose(A, 0, rtol=1e-15)) or np.all(np.isnan(A)):
        if source is not None and target is not None:
            RWED = np.nan
        elif source is None and target is None:
            RWED = np.empty((A.shape[0], A.shape[1]))
            RWED[:] = np.nan
        else:
            RWED = np.array([np.nan]*A.shape[0])
        return RWED

    assert (isinstance(parameter, float) or isinstance(parameter, int)) and parameter > 0
    assert np.all(np.isclose(A.sum(axis=1), 1, rtol=1e-15)) or np.all(np.isclose(np.unique(A.sum(axis=1)), [0, 1], rtol=1e-15)), f"The transition matrix has to be row normalized | A row sums: \n {A.sum(axis=1)} \n A: \n {A} \n (np.all(A = nan)): {(np.all(A = np.nan))}"
    _singular_fundamental_matrix_errors = 0

    if not via_numpy:
        one = eye(A.shape[0], format="csc")
        Z = inv(csc_matrix(one - A * np.exp(-parameter)))
        D = diags(1. / Z.diagonal(), format="csc")
        ZdotD = Z.dot(D).toarray()
        if np.any(ZdotD == 0):
            ZdotD = np.where(ZdotD == 0, 1e-100, ZdotD)  # Substitute zero with low value (10^-100) for subsequent log evaluation
            RWED = -np.log(ZdotD)
            _singular_fundamental_matrix_errors += 1
        else:
            RWED = -np.log(Z.dot(D).toarray())
    else:
        one = np.identity(A.shape[0])
        Z = np.linalg.inv(one - A * np.exp(-parameter))
        D = np.diag(1. / Z.diagonal())
        ZdotD = Z.dot(D).toarray()
        if np.any(ZdotD == 0):
            ZdotD = np.where(ZdotD == 0, 1e-100, ZdotD)
            RWED = -np.log(ZdotD)
            _singular_fundamental_matrix_errors += 1
        else:
            RWED = -np.log(Z.dot(D))

    if source is not None:
        if target is not None:
            RWED = RWED[source, target]
        else:
            RWED = RWED[source, :]
    elif target is not None:
        RWED = RWED[:, target]
    if not return_errors:
        return RWED
    else:
        return RWED, _singular_fundamental_matrix_errors


# Supporting Functions: -----------------------------------------------------------------------------------------
def sum_weighted_path(A, path: list):
    """Sums all edges of a path through a (presumably weighted) network

    Parameters
    ----------
        A: ndarray
            Adjacency Matrix

        path: list of ints
            list of integer indices denoting continuous path through network

    Returns
    -------
        sum: int, float
            sum of all constituent edge weights for a path over A

    Examples
    --------
    >>> A = np.array([
    >>>     [0, 1, 0, 1],
    >>>     [0, 0, 1, 0.5],
    >>>     [1, 0, 0, 0],
    >>>     [0, 0, 0.5, 0],
    >>> ])

    >>> ec.sum_weighted_path(A, [0, 1, 2, 0, 3])
    4
    >>> ec.sum_weighted_path(A, [1, 3, 2, 0])
    2.0
    """
    return sum([A[path[i]][path[i + 1]] for i in range(len(path) - 1)])


def shortest_path_distances(A, source=None, target=None, reversed_directions=False, prenormalized_A=False):
    """Evaluates the distance of the shortest path(s) between two nodes or sets of nodes.
    Simply adds the weighted, shortest path, as evaluated from networkx shortest paths.

    Parameters
    ----------
    A: ndarray
        Adjacency matrix, assuming larger edge weights imply greater distance

    source: int, list of int
       source node, list of source nodes

    target: int, list of int
       target node, list of target nodes

    reversed_directions: bool, default: False
        option to reverse directionality of shortest path calculation

    prenormalized_A: bool, default: False
        set true if A is already scaled between 0, 1 for minor computational saving

    Returns
    -------
    SPD: Shortest paths between source
            If source and target are specified, a float value is returned that specifies the shortest path distance.

            If either source or target is None a numpy array is returned.
            The position corresponds to the node ID.
            shape = (Nnodes,)

            If both are None a numpy array is returned.
            Each row corresponds to the node ID.
            shape = (Nnodes,Nnodes)

    Examples
    --------

    """
    if not prenormalized_A:
        Adj = matrix_normalize(A, row_normalize=not reversed_directions)
    else:
        Adj = A
    Adj = np.array(1 - Adj)  # Preliminary effective distance, converting edge weights from indicating proximity to distances for evaluation by shortest path
    Adj = np.where(Adj == 0, 1e-100, Adj)  # Replacing 0s by ~0s, so network remains connected for shortest path algorithm

    inverted_weights_nx_graph = to_directed(from_numpy_matrix(Adj, create_using=DiGraph))
    if reversed_directions:
        inverted_weights_nx_graph = inverted_weights_nx_graph.reverse(copy=False)

    shortest_paths = shortest_path(inverted_weights_nx_graph, source=source, target=target, weight='weight')
    if source is None:
        if target is None:
            SPD = np.array([[sum_weighted_path(Adj, shortest_paths[s][t]) for s in range(Adj.shape[0])] for t in range(Adj.shape[0])])
        else:
            SPD = np.array([sum_weighted_path(Adj, shortest_paths[s][target]) for s in range(Adj.shape[0])])
    else:
        if target is None:
            SPD = np.array([sum_weighted_path(Adj, shortest_paths[source][t]) for t in range(Adj.shape[0])])
        else:
            SPD = sum_weighted_path(Adj, shortest_paths[source][target])
    return SPD


# Efficiency Coordinates: ---------------------------------------------------------------------------------------
def ave_network_efficiencies(n, ensemble_size: int, efficiency: str):
    """For use in efficiency coordinate normalization,
     returns efficiency coordinate for lattice and random graphs of equal size

     Parameters
     ----------
     n : int
        num nodes in graph

    ensemble_size: int
        number of randomized graphs over which an average will be returned

    efficiency: str
        "routing" or "rout" for routing efficiency
        "diffusion" or "diff" for diffusion efficiency

    Returns
    -------
    tuple : floats
        (lattice average, random graph average)
     """
    if efficiency == "routing" or efficiency == "rout":
        routing_lattice_average = E_rout(A=matrix_normalize(to_numpy_array(grid_graph(dim=[2, int(n / 2)], periodic=True)), row_normalize=True), normalize=False)
        routing_rnd_graph_average = np.mean([E_rout(A=matrix_normalize(to_numpy_array(fast_gnp_random_graph(n=n, p=0.5, seed=i, directed=True)), row_normalize=True), normalize=False) for i in range(ensemble_size)])
        return routing_lattice_average, routing_rnd_graph_average
    if efficiency == "diffusive" or efficiency == "diff":
        diffusive_lattice_average = E_diff(A=matrix_normalize(to_numpy_array(grid_graph(dim=[2, int(n / 2)], periodic=True)), row_normalize=True), normalize=False)
        diffusive_rnd_graph_average = np.mean([E_diff(A=matrix_normalize(to_numpy_array(fast_gnp_random_graph(n=n, p=0.5, seed=i, directed=True)), row_normalize=True), normalize=False) for i in range(ensemble_size)])
        return diffusive_lattice_average, diffusive_rnd_graph_average


def E_rout(A, reversed_directions=False, normalize=True):
    """Routing efficiency, as per _[3]

    Parameters
    ----------
    A: ndarray
        Adjacency matrix

    reversed_directions: bool, default: False
        if true, reverses directionality of directed graph

    normalize: bool, default: True
        if true, normalizes routing efficiency against lattice and randomized graph of equal size

    Returns
    -------
    float :
        Routing Efficiency

    Notes
    ------
    [3]: Efficient behavior of small-world networks
         Latora, Vito, and Massimo Marchiori.
         Physical review letters 87, no. 19 (2001): 198701.
    """
    # To avoid divide by 0 errors, we add the 'average' edge weight 1/self.nodes.shape[1] to the inverted weight shortest paths
    shortest_paths = shortest_path_distances(A=A, source=None, target=None, reversed_directions=reversed_directions)
    n = A.shape[0]

    if normalize:
        routing_lattice_average, routing_rnd_graph_average = ave_network_efficiencies(n=n, ensemble_size=10, efficiency="routing")
        E_routing_base = np.sum(1 / (np.array(shortest_paths) + (1 / n))) / (n * (n - 1))
        return (E_routing_base - routing_lattice_average) / (routing_rnd_graph_average - routing_lattice_average)

    return np.sum(1 / (np.array(shortest_paths) + (1 / n))) / (n * (n - 1))


def E_diff(A, normalize=True):  # Diffusion Efficiency
    """Diffusion efficiency, as per _[3]

    Parameters
    ----------
    A: ndarray
        Adjacency matrix

    normalize: bool, default: True
        if true, normalizes diffusion efficiency against lattice and randomized graph of equal size

    Returns
    -------
    float :
        Diffusion Efficiency

    Notes
    ------
    [3]: Efficient behavior of small-world networks
         Latora, Vito, and Massimo Marchiori.
         Physical review letters 87, no. 19 (2001): 198701.
    """

    Adj_Matrix, n = A, A.shape[0]
    row_sums = Adj_Matrix.sum(axis=1)
    # normalized_A = np.array([Adj_Matrix[node, :] / row_sums[node] for node in range(Adj_Matrix.shape[0])])
    normalized_A = matrix_normalize(A, row_normalize=True)

    if normalize:
        diffusive_lattice_average, diffusive_rnd_graph_average = ave_network_efficiencies(n=n, ensemble_size=10, efficiency="diffusive")
        E_diff_base = np.sum(Random_Walker_Effective_Distance(A=normalized_A)) / (n * (n - 1))
        return (E_diff_base - diffusive_lattice_average) / (diffusive_rnd_graph_average - diffusive_lattice_average)

    return np.sum(Random_Walker_Effective_Distance(A=normalized_A)) / (n * (n - 1))


def network_efficiencies(A, normalize=True):
    """Evaluates diffusion, routing efficiencies according to [3]_.

    Parameters
    ----------
    A: ndarray
        Adjacency matrix

    normalize: bool, default: True
        if true, normalizes diffusion and routing efficiency against lattice and randomized graph of equal size

    Returns
    -------
    tuple of floats:
        Diffusion, routing efficiencies

    Notes
    ------
    [3]: Efficient behavior of small-world networks
         Latora, Vito, and Massimo Marchiori.
         Physical review letters 87, no. 19 (2001): 198701.
    """
    return E_diff(A, normalize=normalize), E_rout(A, normalize=normalize)
