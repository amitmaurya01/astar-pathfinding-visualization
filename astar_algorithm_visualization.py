import numpy as np
import matplotlib.pyplot as plt
from heapq import heapify, heappush, heappop
from collections import defaultdict

# Define the size of the map and the occupancy ratio
# Rows and columns represent the grid, and the occupancy ratio controls the density of obstacles
rows = 40
cols = 60
occupancy_ratio = 0.1

# Randomly generate a binary map where 'True' represents an obstacle and 'False' is a free space
map = np.random.rand(rows, cols) < occupancy_ratio

def get_random_free_node(map):
    """
    Find and return a random free position (not an obstacle) on the map.
    
    Args:
        map (np.ndarray): 2D map array where True is an obstacle and False is a free cell.
    
    Returns:
        tuple: A random (row, col) position that is not an obstacle.
    """
    while True:
        row = np.random.randint(0, rows)
        col = np.random.randint(0, cols)
        if not map[row, col]:  # If the cell is free (not an obstacle)
            return (row, col)

# Get random free positions for start and goal nodes
start_node = get_random_free_node(map)
goal_node = get_random_free_node(map)

def get_neighbours(u):
    """
    Get valid neighboring cells of a given position 'u' considering all 8 possible directions.
    
    Args:
        u (tuple): Current node (row, col) on the map.
    
    Returns:
        list of tuples: A list of tuples where each tuple contains:
                        - cost: The cost to reach the neighbor.
                        - neighbor: The neighboring node (row, col).
    """
    neighbours = []
    
    # Possible movements in 8 directions (right, left, down, up, diagonals)
    for delta in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        cand = (u[0] + delta[0], u[1] + delta[1])
        # Euclidean distance to neighbor and heuristic cost (distance to goal)
        cost = np.sqrt(delta[0]**2 + delta[1]**2) + np.sqrt((goal_node[0] - u[0])**2 +
                                                            (goal_node[1] - u[1])**2)
        # Check if the neighbor is within map boundaries and not an obstacle
        if (cand[0] >= 0 and cand[0] < len(map) and 
            cand[1] >= 0 and cand[1] < len(map[0]) and not map[cand[0]][cand[1]]):
            neighbours.append((cost, cand))
    
    return neighbours

# Initialize priority queue (min-heap) with the start node and distance 0
queue = [(0, start_node)]
heapify(queue)  # Ensure it follows min-heap property

# Dictionary to store the shortest distances to each node, default to infinity
distances = defaultdict(lambda: float("inf"))
distances[start_node] = 0  # Distance to start node is 0

# Set to track visited nodes
visited = {start_node}

# Dictionary to store the parent (previous node) for each node for path reconstruction
parent = {}

# Plot the map, showing start and goal positions
plt.imshow(map)  # Display the occupancy grid map
plt.ion()  # Turn on interactive plotting
plt.plot(start_node[1], start_node[0], 'r+', markersize=16)  # Plot start node as red cross
plt.plot(goal_node[1], goal_node[0], 'y*', markersize=15)  # Plot goal node as yellow star

# A* Search Algorithm Loop
while queue:
    currentdist, v = heappop(queue)  # Get node with the smallest distance
    visited.add(v)  # Mark the current node as visited
    
    # Plot current node being explored in green
    plt.plot(v[1], v[0], 'g*', markersize=10)
    plt.show()
    plt.pause(0.000001)  # Short pause to visualize progress
    
    # If goal is reached, break out of the loop
    if goal_node == v:
        break
    
    # Explore neighbors of the current node
    for costvu, u in get_neighbours(v):
        if u not in visited:
            newcost = distances[v] + costvu  # Calculate the new distance
            # If a shorter path to 'u' is found, update the distance and add to the queue
            if newcost < distances[u]:
                distances[u] = newcost
                heappush(queue, (newcost, u))  # Push the neighbor with its updated cost
                parent[u] = v  # Set the parent for path reconstruction

# Reconstruct the shortest path from start to goal
key = goal_node
path = []

# Trace back from goal to start using the parent dictionary
while key in parent:
    key = parent[key]
    path.insert(0, key)  # Insert nodes in reverse order

# Append the goal node to complete the path
path.append(goal_node)
print(path)

# Visualize the shortest path in red dots
for p in path:
    plt.plot(p[1], p[0], 'r.', markersize=18)
    plt.show()
    plt.pause(0.000001)

# Hold the final plot until closed by the user
plt.show(block=True)
