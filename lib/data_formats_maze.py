import networkx as nx
import random
import numpy as np


#################################################
############### Graph with Wilson ###############
#################################################

def assign_labels(nodes, label_range=(0, 99)):
    """Assign random unique labels to nodes within a specified range."""
    labels = random.sample(range(label_range[0], label_range[1] + 1), len(nodes))
    return {node: label for node, label in zip(nodes, labels)}


def create_tree_with_hops_wilson(total_nodes, num_hops):
    """
    Creates a tree (like an MST) with a specific number of hops between start and end nodes.
    """
    if num_hops >= total_nodes:
        raise ValueError(f"Number of hops must be less than the total number of nodes. \n hops: {num_hops} | nodes: {total_nodes}")

    # Step 1: Create the main path with num_hops
    graph = nx.path_graph(num_hops + 1)  # Creates a linear path with num_hops edges

    # Step 2: Add extra nodes to the tree with Wilson's algorithm (random walk)
    current_nodes = list(graph.nodes())
    new_nodes = list(range(num_hops + 1, total_nodes))
    # wilson's algo
    while new_nodes:
        new_node = new_nodes.pop()
        # random walk to reach graph
        walk = [new_node]
        while walk[-1] not in current_nodes:
            # choose random node from current & new nodes
            random_node = random.choice(current_nodes + new_nodes)
            walk.append(random_node)
            if random_node in new_nodes:
                new_nodes.remove(random_node)
        # add edges
        for i in range(len(walk) - 1):
            graph.add_edge(walk[i], walk[i + 1])
        current_nodes.append(new_node)  # Add the new node to the list of available nodes

    # Step 3: Set the start and end nodes for the main path
    start_node = 0
    end_node = num_hops

    return graph, start_node, end_node


def format_graph_with_hops_and_path_with_shuffle(graph, start_node, end_node):
    """
    Formats the graph with a specific number of hops into the desired string format, 
    and outputs the explicit path from start_node to end_node.
    """
    # Assign random labels to nodes
    node_labels = assign_labels(graph.nodes(), label_range=(1, 99))
    
    # Get the shortest path (in terms of edge count) from start_node to end_node
    shortest_path = nx.shortest_path(graph, source=start_node, target=end_node)

    # Format the path as a string
    path_labels = [node_labels[node] for node in shortest_path]
    path_string = ">".join(map(str, path_labels))
    
    # Format start and end nodes
    start_label = node_labels[start_node]
    end_label = node_labels[end_node]
    start_end_str = f"{start_label}>{end_label}#"

    # Build graph_str with end_node connections at the end
    graph_str = ""
    start_node_str = ""  # Temporary storage for the start_node part
    end_node_str = ""  # Temporary storage for the end_node part
    
    # randomize the order of nodes
    random_nodes = list(graph.nodes())
    random.shuffle(random_nodes)
    for node in random_nodes:
        node_label = node_labels[node]
        # randomize the order of neighbors
        random_neighbors = list(graph.adj[node])
        random.shuffle(random_neighbors)
        neighbor_labels = [node_labels[neighbor] for neighbor in random_neighbors]
        # if node == end_node:
        #     end_node_str = f"{node_label}:" + ",".join(map(str, neighbor_labels)) + "-"
        # elif node == start_node:
        #     start_node_str = f"{node_label}:" + ",".join(map(str, neighbor_labels)) + "-"
        # else:
        #     graph_str += f"{node_label}:" + ",".join(map(str, neighbor_labels)) + "-"
        graph_str += f"{node_label}:" + ",".join(map(str, neighbor_labels)) + "-"

    # Combine everything, placing the end_node last
    graph_str = start_node_str + graph_str + end_node_str

    return start_end_str + graph_str[:-1] + "=", path_string, node_labels




def generate_maze_with_wilson(total_nodes=20, num_hops=3):
    """
    Generates a maze with a specific number of hops between the start and end nodes.
    """
    total_nodes, num_hops = int(total_nodes), int(num_hops)
    graph, start_node, end_node = create_tree_with_hops_wilson(total_nodes, num_hops)
    maze_str, shortest_path, _ = format_graph_with_hops_and_path_with_shuffle(graph, start_node, end_node)

    return maze_str, shortest_path, None

###########################################
############ Graph validatiaon ############
###########################################

def parse_graph_to_dict(graph_string):
    """ Parses the updated graph string format into a dictionary for direct connection lookup. """
    # Extract the start and end nodes
    start_end, connections_part = graph_string.split("#")
    start_node, end_node = map(int, start_end.split(">"))

    # Initialize the graph dictionary
    graph_dict = {}
    
    # Remove the trailing "=" and split by "-"
    connections = connections_part.strip("=").split("-")
    
    for connection in connections:
        # Split each connection into node and its neighbors
        node, neighbors = connection.split(":")
        node = int(node)  # Convert node label to integer
        neighbor_nodes = set(map(int, neighbors.split(","))) if neighbors else set()
        
        # Add node and its neighbors to the dictionary
        graph_dict[node] = neighbor_nodes
    
    return graph_dict, start_node, end_node


def validate_path(graph_dict, path, verify_moves=True, verify_ends=False, start_node=None, end_node=None):
    """ Checks if a given path (sequence of node labels) is valid using a dictionary representation of the graph. """

    try:
        path_labels = list(map(int, path.split(">")))
    except:
        print("="*30)
        print("Invalid path: path must be a string of node labels separated by '>'.")
        print(f"Received: {path}")
        print("="*30)

        return False

    path_labels = list(map(int, path.split(">")))

    if verify_ends:
        # this is the case where verify_maze_nodes is True
        if path_labels[0] != start_node or path_labels[-1] != end_node:
            print("="*30)
            print("Invalid path: path must start and end with the correct nodes.")
            print(f"Received: {path} | Expected: {start_node} -> ... -> {end_node}")
            print("="*30)

            return False

    if verify_moves:
        for i in range(len(path_labels) - 1):
            current_node = path_labels[i]
            next_node = path_labels[i + 1]
            # Check if there is a direct connection in graph_dict
            if next_node not in graph_dict.get(current_node, set()):
                # print(f"Invalid path: {current_node} -> {next_node} | {path}")
                return False
            
    return True


def evaluate_path(graph_dict, path, start_node=None, end_node=None, verify_moves=True, verify_ends=True):
    """ Checks if a given path (sequence of node labels) is valid using a dictionary representation of the graph. """

    format_correct, move_correct, end_correct = True, True, True

    try:
        path_labels = list(map(int, path.split(">")))
    except:
        print("="*30)
        print("Invalid path: path must be a string of node labels separated by '>'.")
        print(f"Received: {path}")
        print("="*30)

        format_correct = False
        end_correct = False
        move_correct = False

        return format_correct, move_correct, end_correct

    path_labels = list(map(int, path.split(">")))

    if verify_ends:
        # this is the case where verify_maze_nodes is True
        if path_labels[0] != start_node or path_labels[-1] != end_node:
            print("="*30)
            print("Invalid path: path must start and end with the correct nodes.")
            print(f"Received: {path} | Expected: {start_node} -> ... -> {end_node}")
            print("="*30)

            end_correct = False

    if verify_moves:
        for i in range(len(path_labels) - 1):
            current_node = path_labels[i]
            next_node = path_labels[i + 1]
            # Check if there is a direct connection in graph_dict
            if next_node not in graph_dict.get(current_node, set()):
                print(f"Invalid path: {current_node} -> {next_node} | {path}")
                
                move_correct = False
            
    return format_correct, move_correct, end_correct




if __name__ == '__main__': 
    print('wilson')
    maze, shortest_path, _ = generate_maze_with_wilson()
    print(maze)
    print(shortest_path)
    print('='*20)

