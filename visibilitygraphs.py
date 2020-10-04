from itertools import combinations
import networkx as nx


def find_visible_edges_to_max(graph, series, left, right):
    if left < right:
        tseries = series[left:right + 1]
        max_value = max(tseries)
        max_index = left + tseries.index(max_value)
        for i in range(left, max_index):
            connect = True
            ya = series[i]
            for j in range(i + 1, max_index):
                yc = series[j]
                if yc > max_value + (ya - max_value) * ((max_index - j) /
                   (max_index - i)):
                    connect = False
                    break
            if connect:
                graph.add_edge(i, max_index)
        for i in range(max_index + 1, right + 1):
            connect = True
            yb = series[i]
            for j in range(max_index + 1, i):
                yc = series[j]
                if yc > yb + (max_value - yb) * ((i - j) / (i - max_index)):
                    connect = False
                    break
            if connect:
                graph.add_edge(max_index, i)
    else:
        max_index = -1

    return max_index


def find_horizontal_visible_edges_to_max(graph, series, left, right):
    if left < right:
        tseries = series[left:right + 1]
        max_value = max(tseries)
        max_index = left + tseries.index(max_value)
        for i in range(left, max_index):
            connect = True
            ya = series[i]
            for j in range(i + 1, max_index):
                yc = series[j]
                if yc > ya:
                    connect = False
                    break
            if connect:
                graph.add_edge(i, max_index)
        for i in range(max_index + 1, right + 1):
            connect = True
            yb = series[i]
            for j in range(max_index + 1, i):
                yc = series[j]
                if yc > yb:
                    connect = False
                    break
            if connect:
                graph.add_edge(max_index, i)
    else:
        max_index = -1

    return max_index


def visibility_graph(series):
    """
    This is based on
    X. Lan, H. Mo, S. Chen, Q. Liu, and Y. Deng, “Fast transformation from time
    series to visibility graphs,” Chaos Interdiscip. J. Nonlinear Sci., 
    vol. 25, no. 8, p. 083105, Aug. 2015, doi: 10.1063/1.4927835.
    """
    g = nx.Graph()

    # add all the nodes
    for i, value in enumerate(series):
        g.add_node(i, mag=value)

    # add visible edges to maximum point
    todo_list = []
    max_index = find_visible_edges_to_max(g, series, 0, len(series) - 1)
    if max_index != -1:
        if max_index == 0:
            todo_list.append((max_index + 1, len(series) - 1))
        elif max_index == (len(series) - 1):
            todo_list.append((0, max_index - 1))
        else:
            todo_list.append((max_index + 1, len(series) - 1))
            todo_list.append((0, max_index - 1))
    else:
        todo_list = []

    # loop until todo_list is empty
    while (todo_list):
        left, right = todo_list.pop()
        max_index = find_visible_edges_to_max(g, series, left, right)
        if max_index != -1:
            if max_index == left:
                todo_list.append((max_index + 1, right))
            elif max_index == right:
                todo_list.append((left, max_index - 1))
            else:
                todo_list.append((max_index + 1, right))
                todo_list.append((left, max_index - 1))

    return g


def horizontal_visibility_graph(series):
    """
    This is based on
    X. Lan, H. Mo, S. Chen, Q. Liu, and Y. Deng, “Fast transformation from time
    series to visibility graphs,” Chaos Interdiscip. J. Nonlinear Sci., 
    vol. 25, no. 8, p. 083105, Aug. 2015, doi: 10.1063/1.4927835.
    """
    g = nx.Graph()

    # add all the nodes
    for i, value in enumerate(series):
        g.add_node(i, mag=value)

    # add visible edges to maximum point
    todo_list = []
    max_index = find_horizontal_visible_edges_to_max(g, series, 0,
                                                     len(series) - 1)
    if max_index != -1:
        if max_index == 0:
            todo_list.append((max_index + 1, len(series) - 1))
        elif max_index == (len(series) - 1):
            todo_list.append((0, max_index - 1))
        else:
            todo_list.append((max_index + 1, len(series) - 1))
            todo_list.append((0, max_index - 1))
    else:
        todo_list = []

    # loop until todo_list is empty
    while (todo_list):
        left, right = todo_list.pop()
        max_index = find_horizontal_visible_edges_to_max(g, series, left,
                                                         right)
        if max_index != -1:
            if max_index == left:
                todo_list.append((max_index + 1, right))
            elif max_index == right:
                todo_list.append((left, max_index - 1))
            else:
                todo_list.append((max_index + 1, right))
                todo_list.append((left, max_index - 1))

    return g


# The following methods are used to calculate features as described in
# G. Zhu, Y. Li, and P. Wen, “Analysis and Classification of Sleep Stages
# Based on Difference Visibility Graphs From a Single-Channel EEG Signal,”
# IEEE J. Biomed. Health Inform., vol. 18, no. 6, pp. 1813–1821, Nov. 2014,
# doi: 10.1109/JBHI.2014.2303991.

def calc_mean_degree(vis_graph):
    sum = 0
    n = 0

    for node, degree in vis_graph.degree:
        sum += degree
        n += 1
    return sum / n


def calc_degree_probabilities(vis_graph, max_degree=11):
    dp_list = [0.0] * (max_degree + 1)
    dh = nx.degree_histogram(vis_graph)
    total = sum(dh)
    dh = [dh[i] / total for i in range(min(max_degree + 1, len(dh)))]

    for i in range(min(max_degree + 1, len(dh))):
        dp_list[i] = dh[i]

    return dp_list
