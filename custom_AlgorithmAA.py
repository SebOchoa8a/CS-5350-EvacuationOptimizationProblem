import heapq
import copy
import time
import networkx as nx
import matplotlib.pyplot as plt

# G (Graph): Adjacency List
# Structure: {Node: [(Neighbor, Time_Weight, Capacity_Ce)]}
G = {
    # Floor 3 (F3)
    'R301_F3': [('H_F3', 0.5, 5)],
    'H_F3': [('R301_F3', 0.5, 5), ('Stairs_F2_F3', 1.0, 5)],
    'Stairs_F2_F3': [('H_F3', 1.0, 5), ('H_F2', 1.0, 5)],
    # Floor 2 (F2)
    'R201_F2': [('H_F2', 0.5, 10)],
    'H_F2': [('R201_F2', 0.5, 10), ('Stairs_F2_F3', 1.0, 5), ('Stairs_F1_F2', 1.0, 15)],
    'Stairs_F1_F2': [('H_F2', 1.0, 15), ('H_F1', 1.0, 15)],
    # Floor 1 (F1)
    'R101_F1': [('H_F1', 0.5, 10)],
    'H_F1': [('R101_F1', 0.5, 10), ('Stairs_F1_F2', 1.0, 15), ('Exit_A', 1.0, 10), ('Exit_B', 1.0, 10), ('Exit_C', 1.0, 10)],
    # Safe Zones / Exits (S)
    'Exit_A': [], 'Exit_B': [], 'Exit_C': []
}

# V (Starting Vertices and Groups):
V_groups = {
    'R301_F3': 3000,  # Group 3
    'R201_F2': 3000,  # Group 2
    'R101_F1': 3000   # Group 1 
}

S_safe_zones = ['Exit_A', 'Exit_B', 'Exit_C']

#Helper Functions
def get_edge_details(graph, u, v):
    """Retrieves weight (time) and capacity (Ce) for a path (u, v)."""
    for neighbor, time_weight, capacity in graph.get(u, []):
        if neighbor == v:
            return time_weight, capacity
    return None, None 

def EdgeEvacuationCompute(u, v, G_state, flow_attempt):
    """
    Calculates the actual flow (F_g,e(t)) and time taken over edge (u, v), modeling 
    capacity constraints and time-based congestion.
    """
    base_time, capacity = get_edge_details(G_state, u, v)

    if base_time is None:
        return 0, float('inf'), 0 
    
    effective_flow = min(flow_attempt, capacity)
    time_cost = base_time
    
    if effective_flow > 0 and effective_flow == capacity:
        time_cost *= 2.0  # Congestion penalty

    residual_capacity = capacity - effective_flow
    return effective_flow, time_cost, residual_capacity

def update_edge_capacity(graph, u, v, new_capacity):
    """Helper function to update the graph state with residual capacity."""
    if u not in graph:
        return
    
    for i, (neighbor, time, old_cap) in enumerate(graph[u]):
        if neighbor == v:
            graph[u][i] = (neighbor, time, new_capacity)
            break
            
    for i, (neighbor, time, old_cap) in enumerate(graph.get(v, [])):
        if neighbor == u:
            graph[v][i] = (neighbor, time, new_capacity)
            break

#Custom dijkstra_shortest_path algorithm
def dijkstra_shortest_path(graph, start_node, end_nodes, current_edge_usage):
    """
    Finds the shortest path from a start_node to *any* of the end_nodes.
    This version is now "capacity-aware" and "congestion-aware".
    """
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        for n, _, _ in neighbors:
            all_nodes.add(n)
    for n in end_nodes:
        all_nodes.add(n)
    
    distances = {node: float('inf') for node in all_nodes}
    distances[start_node] = 0
    previous_nodes = {}
    pq = [(0, start_node)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node in end_nodes:
            path = []
            while current_node in previous_nodes:
                path.insert(0, current_node)
                current_node = previous_nodes[current_node]
            path.insert(0, start_node)
            return current_distance, path
            
        if current_distance > distances[current_node]:
            continue
            
        for neighbor, time_weight, max_capacity in graph.get(current_node, []):
            edge = (current_node, neighbor)
            
            # Check if the path is currently full
            if current_edge_usage.get(edge, 0) >= max_capacity:
                continue

            congestion_penalty = current_edge_usage.get(edge, 0)
            distance = current_distance + time_weight + congestion_penalty
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
                
    return float('inf'), []

def EvacuationOptimization(G_initial, V_groups_initial, S_safe_zones):
    """
    Runs a time-step simulation to model evacuation flow.
    """
    
    # --- INITIALIZATION ---
    sim_time = 0.0
    total_evacuees_saved = 0
    total_population = sum(V_groups_initial.values())
    
    population_state = copy.deepcopy(V_groups_initial)
    in_transit = [] 
    edge_usage = {}
    
    #starts with farthest group first
    nodes_to_process = [
        'R301_F3', 'H_F3',         # Floor 3
        'Stairs_F2_F3', 'R201_F2', # Floor 2
        'H_F2', 'Stairs_F1_F2',    # Floor 1
        'R101_F1', 'H_F1'
    ]
    
    print(f"--- Starting Simulation (Time: {sim_time}) ---")
    print(f"Total Evacuees to save: {total_population}")
    
    # Run for a max of 1000 steps to prevent infinite loops
    for _ in range(1000):
        
        # --- 1. ARRIVALS ---
        for arrival in list(in_transit):
            arrival_time, dest_node, group_id, num_people, edge_used = arrival
            
            if sim_time >= arrival_time:
                in_transit.remove(arrival)
                edge_usage[edge_used] = edge_usage.get(edge_used, 0) - num_people
                
                if dest_node in S_safe_zones:
                    total_evacuees_saved += num_people
                    print(f"  > [Time {sim_time:.1f}] {num_people} from {group_id} arrived at {dest_node}!")
                else:
                    population_state[dest_node] = population_state.get(dest_node, 0) + num_people
                    print(f"  > [Time {sim_time:.1f}] {num_people} from {group_id} arrived at intermediate node {dest_node}")

        #Check if simulation is done
        if total_evacuees_saved == total_population:
             print(f"\n--- [Time {sim_time:.1f}] All evacuees are safe! ---")
             break # Exit the main loop
             
        # --- 2. DEPARTURES ---
        movement_happened = False
        
        for start_node in nodes_to_process:
            pop_to_move = population_state.get(start_node, 0)
            
            if pop_to_move > 0:
                _, path = dijkstra_shortest_path(G_initial, start_node, S_safe_zones, edge_usage)
                
                if path:
                    u, v = path[0], path[1]
                    edge = (u, v)
                    travel_time, max_capacity = get_edge_details(G_initial, u, v)
                    available_capacity = max_capacity - edge_usage.get(edge, 0)
                    
                    flow_to_send = min(pop_to_move, available_capacity)
                    
                    if flow_to_send > 0:
                        movement_happened = True
                        arrival_t = sim_time + travel_time
                        
                        original_group_id = start_node if start_node in V_groups_initial else f"{start_node} (Group)"
                        in_transit.append((arrival_t, v, original_group_id, flow_to_send, edge))
                        
                        population_state[start_node] -= flow_to_send
                        edge_usage[edge] = edge_usage.get(edge, 0) + flow_to_send

                        print(f"  > [Time {sim_time:.1f}] {flow_to_send} from {start_node} are moving to {v} (ETA: {arrival_t:.1f})")

        if not movement_happened and not in_transit:
            print(f"\n--- [Time {sim_time:.1f}] No further movement possible. ---")
            break 
            
        sim_time += 0.5 # Advance time by 30 seconds
    
    remaining_evacuees = sum(population_state.values())

    return {
        "Total_Evacuees_Saved": total_evacuees_saved,
        "Evacuation_Time_T": sim_time,
        "Remaining_At_Start": remaining_evacuees
    }

# graph function
def visualize_graph(G_initial, V_groups, S_safe_zones):
    """
    Creates a static plot of the evacuation graph using NetworkX and Matplotlib.
    """
    # Create a directed graph object
    G_nx = nx.DiGraph()

    # Add all edges with their attributes (time and capacity)
    for u, neighbors in G_initial.items():
        for v, time, capacity in neighbors:
            G_nx.add_edge(u, v, time=time, capacity=capacity)

    node_colors = []
    labels = {}
    
    all_nodes = set(G_nx.nodes())
    for node in V_groups.keys():
        all_nodes.add(node)
    for node in S_safe_zones:
        all_nodes.add(node)
            
    for node in all_nodes:
        labels[node] = node # Use the node name as its label
        if node in V_groups:
            node_colors.append('skyblue') # Starting points
        elif node in S_safe_zones:
            node_colors.append('lightgreen') # Safe zones
        else:
            node_colors.append('lightgray') # Intermediate nodes

    edge_labels = {}
    for u, v, data in G_nx.edges(data=True):
        edge_labels[(u, v)] = f"T:{data['time']}, C:{data['capacity']}"
    
    # --- Draw the graph ---
    
    pos = nx.spring_layout(G_nx, k=2.5, iterations=50) 
    
    plt.figure(figsize=(16, 10))
    
    # Draw the nodes, edges, and labels
    nx.draw_networkx_nodes(G_nx, pos, nodelist=all_nodes, node_color=node_colors, node_size=1500, alpha=0.9)
    nx.draw_networkx_edges(G_nx, pos, edge_color='gray', arrowstyle='->', arrowsize=15)
    nx.draw_networkx_labels(G_nx, pos, labels=labels, font_size=8, font_weight='bold')
    nx.draw_networkx_edge_labels(G_nx, pos, edge_labels=edge_labels, font_size=7, font_color='red')
    
    plt.title("UTEP CS Building Evacuation Graph", size=15)
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    
    # Start total timer
    total_start_time = time.time()
    
    print("Displaying evacuation graph...")
    visualize_graph(G, V_groups, S_safe_zones) # You can uncomment this to see the graph
    
    # --- Run the simulation ---
    final_plan_results = EvacuationOptimization(G, V_groups, S_safe_zones) 
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "="*30)
    print("--- FINAL EVACUATION RESULTS ---")
    print(f"Total Evacuees Saved: {final_plan_results['Total_Evacuees_Saved']}")
    print(f"Evacuation Time (T): {round(final_plan_results['Evacuation_Time_T'], 2)} minutes (Time the last evacuee reached safety)")
    print(f"Evacuees Left Behind: {final_plan_results['Remaining_At_Start']}")
    print("="*30)
    print(f"Total Execution Time: {total_duration:.4f} seconds")
    print("="*30)