from __future__ import print_function, division
import pylp
import logging

logger = logging.getLogger(__name__)


def match_components(
        nodes_x, nodes_y,
        edges_xy,
        node_labels_x, node_labels_y,
        edge_conflicts=None,
        max_edges=None,
        optimality_gap=0.0,
        time_limit=None):

    '''Match nodes from X to nodes from Y by selecting candidate edges x <-> y,
    such that the split/merge error induced from the labels for X and Y is
    minimized.

    Example::

        X:     Y:

        1      a
        |      |
        2      b
        |       \
        3      h c
        |      | |
        4    C i d
        |      | |
        5      j e
        |       /
        6      f
        |      |
        7      g

        A      B

    1-7: nodes in X labelled A; a-g: nodes in Y labelled B; h-j: nodes in Y
    labelled C.

    Assuming that all nodes in X can be matched to all nodes in Y in the same
    line (``edges_xy`` would be (1, a), (2, b), (3, h), (3, c), and so on), the
    solution would be to match:

        1 - a
        2 - b
        3 - c
        4 - d
        5 - e
        6 - f
        7 - g

    h, i, and j would remain unmatched, since matching them would incur a split
    error of A into B and C.

    Args:

        nodes_x, nodes_y (array-like of ``int``):

            A list of IDs of set X and Y, respectively.

        edges_xy (array-like of tuple):

            A list of tuples ``(id_x, id_y)`` of matching edges to chose from.

        node_labels_x, node_labels_y (``dict``):

            A dictionary from IDs to labels.

    Returns:

        (label_matches, node_matches, num_splits, num_merges, num_fps, num_fns)

        ``label_matches``: A list of tuples ``(label_x, label_y)`` of labels
        that got matched.

        ``node_matches``: A list of tuples ``(id_x, id_y)`` of nodes that got
        matched. Subset of ``edges_xy``.

        ``num_splits``, ``num_merges``, ...: The number of label splits,
        merges, false positives (unmatched in X), and false negatives
        (unmatched in Y).
    '''

    num_vars = 0

    # add "no match in X" and "no match in Y" dummy nodes
    no_match_node = max(nodes_x + nodes_y) + 1
    no_match_label = max(max(node_labels_x.keys()), max(node_labels_y.keys())) + 1

    node_labels_x = dict(node_labels_x)
    node_labels_y = dict(node_labels_y)
    node_labels_x.update({no_match_node: no_match_label})
    node_labels_y.update({no_match_node: no_match_label})

    labels_x = set(node_labels_x.values())
    labels_y = set(node_labels_y.values())

    # add additional edges to dummy nodes
    edges_xy += [ (n, no_match_node) for n in nodes_x ]
    edges_xy += [ (no_match_node, n) for n in nodes_y ]

    # create indicator for each matching edge
    edge_indicators = {}
    edges_by_node_x = {}
    edges_by_node_y = {}
    for edge in edges_xy:
        edge_indicators[edge] = num_vars
        num_vars += 1
        u, v = edge
        if u not in edges_by_node_x:
            edges_by_node_x[u] = []
        if v not in edges_by_node_y:
            edges_by_node_y[v] = []
        edges_by_node_x[u].append(edge)
        edges_by_node_y[v].append(edge)

    # Require that each node matches to 1<=n<=max_edges

    constraints = pylp.LinearConstraints()
    conflicts = []
    for nodes, edges_by_node in zip(
            [nodes_x, nodes_y], [edges_by_node_x, edges_by_node_y]):

        for node in nodes:

            if node == no_match_node:
                continue

            constraint_low = pylp.LinearConstraint()
            constraint_high = pylp.LinearConstraint()
    
            for edge in edges_by_node[node]:
                constraint_low.set_coefficient(edge_indicators[edge], 1)
                constraint_high.set_coefficient(edge_indicators[edge], 1)

                if not no_match_node in edge:
                    conflict = pylp.LinearConstraint()
                    conflict.set_coefficient(edge_indicators[edge], 1)

                    potential_conflict = tuple([edge[0], no_match_node])
                    if not potential_conflict in conflicts:
                        conflict_edge = potential_conflict
                    else:
                        conflict_edge = tuple([no_match_node, edge[1]]) 

                    conflicts.append(conflict_edge)                
                    conflict.set_coefficient(edge_indicators[conflict_edge], 1)
                    conflict.set_relation(pylp.Relation.LessEqual)
                    conflict.set_value(1)
                    constraints.add(conflict)

            constraint_low.set_relation(pylp.Relation.GreaterEqual)
            constraint_low.set_value(1)
            constraints.add(constraint_low)

            if max_edges is not None:
                constraint_high.set_relation(pylp.Relation.LessEqual)
                constraint_high.set_value(max_edges)
                constraints.add(constraint_high)


    # add indicators for label matches
    label_indicators = {}
    edges_by_label_pair = {}
    binary_label_indicators = {}

    for edge in edges_xy:

        label_pair = node_labels_x[edge[0]], node_labels_y[edge[1]]

        if label_pair not in label_indicators:
            label_indicators[label_pair] = num_vars
            num_vars += 1
            binary_label_indicators[label_pair] = num_vars
            num_vars += 1

        if label_pair not in edges_by_label_pair:
            edges_by_label_pair[label_pair] = []
        edges_by_label_pair[label_pair].append(edge)

    label_indicators[(no_match_label, no_match_label)] = num_vars
    num_vars += 1
    binary_label_indicators[(no_match_label, no_match_label)] = num_vars
    num_vars += 1

    # couple integer label indicators to edge indicators
    for label_pair, edges in edges_by_label_pair.items():
        constraint = pylp.LinearConstraint()
        constraint.set_coefficient(label_indicators[label_pair], 1)
        for edge in edges:
            constraint.set_coefficient(edge_indicators[edge], -1)
        constraint.set_relation(pylp.Relation.Equal)
        constraint.set_value(0)
        constraints.add(constraint)


    # Couple binary label indicators to integer label indicators
    for label_pair, label_indicator in label_indicators.items():
        constraint1 = pylp.LinearConstraint()
        constraint1.set_coefficient(binary_label_indicators[label_pair], 1) 
        constraint1.set_coefficient(label_indicator, -1)
        constraint1.set_relation(pylp.Relation.LessEqual)
        constraint1.set_value(0)

        constraint2 = pylp.LinearConstraint()
        if not label_pair == (no_match_label, no_match_label):
            constraint2.set_coefficient(binary_label_indicators[label_pair], len(edges_by_label_pair[label_pair]))
            constraint2.set_coefficient(label_indicator, -1)
            constraint2.set_relation(pylp.Relation.GreaterEqual)
            constraint2.set_value(0)

        constraints.add(constraint1)
        constraints.add(constraint2)


    if edge_conflicts is not None:
        for conflict in edge_conflicts:
            constraint = pylp.LinearConstraint()
            for edge in conflict:
                constraint.set_coefficient(edge_indicators[tuple(edge)], 1)
           
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(1)
            constraints.add(constraint)


    # pin binary no-match pair indicator to 1
    constraint = pylp.LinearConstraint()
    no_match_indicator = binary_label_indicators[(no_match_label, no_match_label)]
    constraint.set_coefficient(no_match_indicator, 1)
    constraint.set_relation(pylp.Relation.Equal)
    constraint.set_value(1)
    constraints.add(constraint)

    # set objective
    objective = pylp.LinearObjective(num_vars)
    for label_pair, indicator in label_indicators.items():
        if not (no_match_label in label_pair):
            objective.set_coefficient(indicator, len(label_indicators) + 1)
            objective.set_coefficient(binary_label_indicators[label_pair], -1)

    objective.set_sense(pylp.Sense.Maximize)

    # solve
    logger.debug("Added %d constraints", len(constraints))
    for i in range(len(constraints)):
        logger.debug(constraints[i])

    logger.debug("Creating quadratic solver")
    solver = pylp.create_linear_solver(pylp.Preference.Any)
    variable_types = pylp.VariableTypeMap()
    for label_pair, indicator in label_indicators.items():
        variable_types[indicator] = pylp.VariableType.Integer



    if optimality_gap is not None:
        solver.set_optimality_gap(optimality_gap, True)

    if time_limit is not None:
        solver.set_timeout(time_limit)

    logger.debug("Initializing solver with %d variables", num_vars)
    solver.initialize(num_vars, pylp.VariableType.Binary, variable_types)
    solver.set_num_threads(1)

    logger.debug("Setting objective")
    solver.set_objective(objective)

    logger.debug("Setting constraints")
    solver.set_constraints(constraints)

    logger.debug("Solving...")
    solution, message = solver.solve()

    logger.debug("Solver returned: %s", message)
    if 'NOT' in message:
        print("Suboptimal solution found.")

    # get label matches
    total_value = 0
    label_matches = []
    for label_pair, label_indicator in binary_label_indicators.items():
        if True:
            if solution[label_indicator] > 0.5:
                label_matches.append(label_pair)

    # get node matches
    node_matches = [
        e
        for e in edges_xy
        if solution[edge_indicators[e]] > 0.5 and no_match_node not in e
    ]

    # get macroscopic errors counts
    print("labels_x", labels_x)
    print("labels_y", labels_y)
    print("label_matches", label_matches)
    splits = len(label_matches) - len(labels_x)
    merges = len(label_matches) - len(labels_y)

    fps = 0
    fns = 0
    for label_pair, label_indicator in binary_label_indicators.items():
        if label_pair[0] == no_match_label:
            fps += (solution[label_indicator] > 0.5)
        if label_pair[1] == no_match_label:
            fns += (solution[label_indicator] > 0.5)

    fps -= 1
    fns -= 1

    splits -= fps
    merges -= fns

    return (label_matches, node_matches, splits, merges, fps, fns)
