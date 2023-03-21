# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Name> Trevor Wai
<Class> Section 1
<Date> 3/20/23
"""

import numpy as np
import scipy.linalg as la
import networkx as nx
from itertools import combinations as cb

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        self.n = len(A)

        #Labes for the n nodes of the graph
        if labels == None:
            self.labels = list(np.arange(self.n))
        else:
            self.labels = labels
        
        if len(self.labels) != self.n:
            raise ValueError('The number of labels is not equal to the number of nodes in the graph.')
        #Removes Sinks
        A[:, (np.sum(A, axis=0) == 0)] = 1
        #Normalizes
        self.A = A / np.sum(A, axis=0)


    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #Compute the Pagerank vector
        p = la.solve(np.eye(self.n) - epsilon * self.A, (1-epsilon) * np.ones(self.n) / self.n)
        #Maps the labesl to the PageRank
        return dict(zip(self.labels, p))

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        E = np.ones((self.n, self.n))
        #Computes the PageRank vector
        B = epsilon * self.A + ((1-epsilon) / self.n) * E
        eigvals, eigvects = la.eig(B)
        u = eigvects[:,np.argmax(np.real(eigvals))]
        #Maps the labels to the Pagerank
        return dict(zip(self.labels, u/sum(u)))

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #Initial
        p0 = np.ones(self.n) / self.n
        #Compute the PageRank Vector
        for i in range(maxiter):
            p = epsilon * self.A @ p0 + (1-epsilon) * np.ones(self.n) / self.n
            if sum(abs(p-p0)) < tol:
                break
            p0 = p
        #Maps the labels to the PageRank
        return dict(zip(self.labels,p))


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    #Construct a sorted list of labels based on the PageRank vector
    return sorted(d, key=d.get, reverse=True)


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    with open(filename, 'r') as infile:
        data = infile.read().strip()
        #Creates a list of all the IDs
        layout = data.replace('\n', '/').split('/')
        unique = sorted(set(layout))
        #Maps the ID to the index
        web = dict(zip(unique, range(len(unique))))
        A = np.zeros((len(unique), len(unique)))

        for i in data.split('\n'):
            i = i.split('/')
            for site in i[1:]:
                #Stores the sites link to the others
                A[web[site], web[i[0]]] = 1

    dg = DiGraph(A, web.keys())

    return get_ranks(dg.itersolve(epsilon))
        


# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    with open(filename, 'r') as infile:
        data = infile.read().strip()[13:]
        #Creates list of teams
        layout = data.replace('\n', ',').split(',')
        unique = sorted(set(layout))
        #Maps teams to index
        team = dict(zip(unique, range(len(unique))))
        A = np.zeros((len(unique), len(unique)))

        for i in data.split('\n'):
            i = i.split(',')
            #Store which teams lost
            A[team[i[0]], team[i[1]]] += 1
    
    dg = DiGraph(A, team.keys())
    return get_ranks(dg.itersolve(epsilon))


# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    dg = nx.DiGraph()

    with open(filename, 'r', encoding='utf-8') as infile:

        lines = infile.readlines()

        for line in lines:
            #Formatting
            line = line.strip().split('/')[1:]
            #Get all pairs
            combs = cb(line, 2)

            for a, b in combs:
                #Adds edges within graph
                if dg.has_edge(b,a):
                    dg[b][a]['weight'] += 1
                else:
                    dg.add_edge(b, a, weight=1)

        return get_ranks(nx.pagerank(dg, alpha=epsilon))