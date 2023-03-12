import re
import numpy as np
import networkx as nx


def normalize(text):
    # equivalent
    return re.sub('[ \t]{2,}', ' ', text).strip()


def split_into_lines(text):
    lines = re.split('\n', text)
    lines = [normalize(_) for _ in lines]  # normalize
    lines = [_ for _ in lines if _.strip()]  # remove empty strings and normalize
    return lines


label_length = 20


def create_graph(matrix, texts, threshold=0):
    A_nodes, B_nodes = matrix.shape[0], matrix.shape[1]
    print(f'nodes: {A_nodes} from A, {B_nodes} from A , {A_nodes + B_nodes} total')

    nx_graph = nx.Graph()
    # nodes
    # matrix
    for i in range(A_nodes):
        global_i = i
        text = texts[0][i]
        nx_graph.add_node(global_i, size=20, group=1,
                          label=text[:label_length] + '...', title=text)
    for j in range(B_nodes):
        global_j = j + A_nodes  # avoid overlapping nodes. it's not an adjacency matrix
        text = texts[1][j]
        nx_graph.add_node(global_j, size=20, group=2,
                          label=text[:label_length] + '...', title=text)

    # text chain: chains items into a sequence graph
    for i in range(1, A_nodes):
        global_i = i
        nx_graph.add_edge(global_i - 1, global_i, weight=1)

    for j in range(1, B_nodes):
        global_j = j + A_nodes  # avoid overlapping nodes. it's not an adjacency matrix
        nx_graph.add_edge(global_j - 1, global_j, weight=1)

    nodes_indices = list(nx_graph.nodes)
    print('nodes indices: ', nodes_indices)
    print('nodes from A: ', nodes_indices[:A_nodes])
    print('nodes from B: ', nodes_indices[A_nodes:])

    # START NODE
    start_node_index = A_nodes + B_nodes
    nx_graph.add_node(start_node_index, size=25, group=0, label='START')
    nx_graph.add_edge(start_node_index, 0)
    nx_graph.add_edge(start_node_index, A_nodes)
    # nx_graph.add_edge(0, start_node_index)
    # nx_graph.add_edge(A_nodes, start_node_index)

    # edges
    for i in range(A_nodes):
        for j in range(B_nodes):
            if matrix[i, j]:
                global_j = j + A_nodes
                nx_graph.add_edge(i, global_j, weight=int(5 * matrix[i, j]))

    return nx_graph


def comparison_matrix(texts, comparison_function):
    # creates a cartesian product matrix from two texts already divide into units
    # e.g. units: lines <- paragraphs,  sentences
    # rows from argument index 0, columns from argument index 1
    matrix = np.zeros((len(texts[0]), len(texts[1])))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            pair = texts[0][i], texts[1][j]
            metric = comparison_function(*pair)
            matrix[i, j] = metric
            # print(metric, pair)

    return matrix


def tokens_jaccard_similarity(a, b):
    # convert to set
    a = set(a.split())
    b = set(b.split())
    # calucate jaccard similarity
    return float(len(a.intersection(b))) / len(a.union(b))


def print_comparison(boolean_matrix, texts):
    print(boolean_matrix)

    print("""
    COMPARISON between A and B:
        - B deletions
        + B additions
        ? ambiguity (multiple edges pointing to the same node)
    """)

    # A: present in A, absent in B
    pa = np.count_nonzero(boolean_matrix, axis=1)
    for index, count in enumerate(pa):
        if count == 0:
            print('-', texts[0][index])
        elif count > 1:
            print('?:', texts[0][index])

    pa = np.count_nonzero(boolean_matrix, axis=0)
    for index, count in enumerate(pa):
        if count == 0:
            print('+', texts[1][index])
        elif count > 1:
            print('?:', texts[1][index])

    # analyze crossing
