import comatch
#import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger('comatch').setLevel(logging.DEBUG)

if __name__ == "__main__":

    nodes_x = list(range(1, 8))
    nodes_y = list(range(101, 111))
    edges_xy = [
        (1, 101),
        (2, 102),
        (3, 103),
        (3, 108),
        (4, 104),
        (4, 109),
        (5, 105),
        (5, 110),
        (6, 106),
        (7, 107),
    ]
    node_labels_x = { n: 1 for n in nodes_x }
    node_labels_y = { n: 2 for n in nodes_y }
    node_labels_y[108] = 3
    node_labels_y[109] = 3
    node_labels_y[110] = 3

    label_matches, node_matches, splits, merges, fps, fns = comatch.match_components(
        nodes_x, nodes_y,
        edges_xy,
        node_labels_x, node_labels_y)

    print(node_matches)
    print("splits: %d"%splits)
    print("merges: %d"%merges)
    print("fps   : %d"%fps)
    print("fns   : %d"%fns)

    # the other way around
    label_matches, node_matches, splits, merges, fps, fns = comatch.match_components(
        nodes_y, nodes_x,
        [ (v, u) for (u, v) in edges_xy ],
        node_labels_y, node_labels_x)

    print(node_matches)
    print("splits: %d"%splits)
    print("merges: %d"%merges)
    print("fps   : %d"%fps)
    print("fns   : %d"%fns)

    # test edge costs

    nodes_x = list(range(1, 8))
    nodes_y = list(range(101, 111))
    edges_xy = [
        (1, 101),
        (1, 102),
        (2, 101),
        (2, 102),
        (3, 103),
        (3, 108),
        (4, 104),
        (4, 109),
        (5, 105),
        (5, 110),
        (6, 106),
        (7, 107),
    ]
    edge_costs = [ 10, 1, 1, 10, 10, 1, 10, 1, 10, 1, 1, 1 ]
    node_labels_x = { n: 1 for n in nodes_x }
    node_labels_y = { n: 2 for n in nodes_y }
    node_labels_y[108] = 3
    node_labels_y[109] = 3
    node_labels_y[110] = 3

    label_matches, node_matches, splits, merges, fps, fns = comatch.match_components(
        nodes_x, nodes_y,
        edges_xy,
        node_labels_x, node_labels_y,
        edge_costs=edge_costs)

    print(node_matches)
    print("splits: %d"%splits)
    print("merges: %d"%merges)
    print("fps   : %d"%fps)
    print("fns   : %d"%fns)

    # test many-to-many matching

    nodes_x = [1, 2, 3, 4]
    nodes_y = [10, 20, 30]
    edges_xy = [
        (1, 10),
        (2, 20),
        (3, 20),
        (4, 30)
    ]
    node_labels_x = { 1: 1, 2: 1, 3: 1, 4: 1 }
    node_labels_y = { 10: 10, 20: 10, 30: 10 }

    label_matches, node_matches, splits, merges, fps, fns = comatch.match_components(
        nodes_x, nodes_y,
        edges_xy,
        node_labels_x, node_labels_y,
        max_edges=3)

    print(node_matches)
    print("splits: %d"%splits)
    print("merges: %d"%merges)
    print("fps   : %d"%fps)
    print("fns   : %d"%fns)

    # test edge costs

    n = 100
    nodes_x = list(range(n + 10))
    nodes_y = list(range(n))
    edges_xy = [
        (i, j)
        for i in range(n)
        for j in range(n)
    ]
    edge_costs = [
        np.linalg.norm(np.array([0, i]) - np.array([1, j]))
        for i in range(n)
        for j in range(n)
    ]
    node_labels_x = { n: 1 for n in nodes_x }
    node_labels_y = { n: 1 for n in nodes_y }

    label_matches, node_matches, splits, merges, fps, fns = comatch.match_components(
        nodes_x, nodes_y,
        edges_xy,
        node_labels_x, node_labels_y,
        edge_costs=edge_costs)

    print(node_matches)
    for (i, j) in node_matches:
        assert i==j
    print("splits: %d"%splits)
    print("merges: %d"%merges)
    print("fps   : %d"%fps)
    print("fns   : %d"%fns)
