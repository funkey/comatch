import comatch
import logging

logging.basicConfig(level=logging.INFO)
#logging.getLogger('comatch').setLevel(logging.DEBUG)

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
