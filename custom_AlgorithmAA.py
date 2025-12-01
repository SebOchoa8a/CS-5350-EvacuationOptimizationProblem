import heapq
import copy
import time
import networkx as nx
import matplotlib.pyplot as plt

building_graph = {
    'R301_F3': [('H_F3', 0.5, 5)],
    'H_F3': [('R301_F3', 0.5, 5), ('Stairs_F2_F3', 1.0, 5)],
    'Stairs_F2_F3': [('H_F3', 1.0, 5), ('H_F2', 1.0, 5)],
    
    'R201_F2': [('H_F2', 0.5, 10)],
    'H_F2': [('R201_F2', 0.5, 10), ('Stairs_F2_F3', 1.0, 5), ('Stairs_F1_F2', 1.0, 15)],
    'Stairs_F1_F2': [('H_F2', 1.0, 15), ('H_F1', 1.0, 15)],
    
    'R101_F1': [('H_F1', 0.5, 10)],
    'H_F1': [('R101_F1', 0.5, 10), ('Stairs_F1_F2', 1.0, 15), ('Exit_A', 1.0, 10), ('Exit_B', 1.0, 10), ('Exit_C', 1.0, 10)],
    
    'Exit_A': [], 'Exit_B': [], 'Exit_C': []
}

starting_groups = {
    'R301_F3': {'pop': 100, 'delay': 1.0},
    'R201_F2': {'pop': 100, 'delay': 4.0},
    'R101_F1': {'pop': 100, 'delay': 1.0}
}

exit_points = ['Exit_A', 'Exit_B', 'Exit_C']

def get_edge_details(graph, u, v):
    """Retrieves weight (time) and capacity (Ce) for a path (u, v)."""
    for neighbor, time_weight, capacity in graph.get(u, []):
        if neighbor == v:
            return time_weight, capacity
    return None, None 

def EdgeEvacuationCompute(u, v, graph_state, flow_attempt):
    """
    Calculates the actual flow and time taken over edge (u, v), modeling 
    capacity constraints.
    """
    base_time, capacity = get_edge_details(graph_state, u, v)

    if base_time is None:
        return 0, float('inf'), 0 
    
    # Capacity Limitation
    effective_flow = min(flow_attempt, capacity)
    time_cost = base_time
    
    # Congestion Penalty
    if effective_flow > 0 and effective_flow == capacity:
        time_cost *= 2.0 

    residual_capacity = capacity - effective_flow
    return effective_flow, time_cost, residual_capacity

def update_edge_capacity(graph, u, v, new_capacity):
    if u not in graph: return
    for i, (neighbor, time, old_cap) in enumerate(graph[u]):
        if neighbor == v:
            graph[u][i] = (neighbor, time, new_capacity)
            break
    for i, (neighbor, time, old_cap) in enumerate(graph.get(v, [])):
        if neighbor == u:
            graph[v][i] = (neighbor, time, new_capacity)
            break

def dijkstra_shortest_path(graph, start, exits, edge_usage, routing_mode='dynamic'):
    """
    Finds the shortest path considering current congestion.
    """
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        for n, _, _ in neighbors: all_nodes.add(n)
    for n in exits: all_nodes.add(n)
    
    dist = {node: float('inf') for node in all_nodes}
    dist[start] = 0
    prev = {}
    pq = [(0, start)]
    
    while pq:
        curr_dist, node = heapq.heappop(pq)
        
        if node in exits:
            path = []
            while node in prev:
                path.insert(0, node)
                node = prev[node]
            path.insert(0, start)
            return curr_dist, path
            
        if curr_dist > dist[node]:
            continue
            
        for neighbor, travel_time, max_cap in graph.get(node, []):
            edge = (node, neighbor)
            
            if edge_usage.get(edge, 0) >= max_cap:
                continue

            # Add penalty for congestion to guide decision making
            penalty = edge_usage.get(edge, 0)
            d = curr_dist + travel_time + penalty
            
            if d < dist[neighbor]:
                dist[neighbor] = d
                prev[neighbor] = node
                heapq.heappush(pq, (d, neighbor))
                
    return float('inf'), []

def EvacuationOptimization(graph, groups, exits, time_limit=None, target_evacuees=None, stop_on_target=False, verbose=False):
    
    t = 0.0
    saved = 0
    target_time = None
    
    # Calculate total population from the new dictionary structure
    total_pop = sum(group['pop'] for group in groups.values())
    
    pop_state = copy.deepcopy(groups)
    
    in_transit = [] 
    edge_flow = {}
    
    process_order = [
        'R301_F3', 'H_F3',         
        'Stairs_F2_F3', 'R201_F2', 
        'H_F2', 'Stairs_F1_F2',    
        'R101_F1', 'H_F1'
    ]
    
    if verbose:
        print(f"--- Starting Simulation (Time: {t}) ---")
        if time_limit: print(f"Scenario: Fire! Time Limit: {time_limit} minutes")
        if target_evacuees: print(f"Scenario: Save {target_evacuees} people ASAP")
        print(f"Total Evacuees: {total_pop}")
    
    for _ in range(2000):
        
        # Check Time Limit
        if time_limit and t > time_limit:
            if verbose:
                print(f"\n!!! TIME LIMIT REACHED ({time_limit} min) !!!")
                print("The building is no longer safe.")
            break

        # --- 1. ARRIVALS ---
        for arrival in list(in_transit):
            # unpack the tuple including the delay factor used for calculation
            eta, dest, group_id, count, edge_used, speed_mod = arrival
            
            if t >= eta:
                in_transit.remove(arrival)
                edge_flow[edge_used] = edge_flow.get(edge_used, 0) - count
                
                if dest in exits:
                    saved += count
                    if verbose: print(f"  > [Time {t:.1f}] {count} from {group_id} arrived at {dest}!")
                    
                    # Check Target
                    if target_evacuees and saved >= target_evacuees:
                        if target_time is None:
                            target_time = t
                            if verbose: print(f"  *** TARGET REACHED: {saved} saved at Time {t:.1f} ***")
                        
                        if stop_on_target:
                            if verbose: print(f"\n--- [Time {t:.1f}] Target of {target_evacuees} reached. Stopping. ---")
                            pass 

                else:
                    # Merge population into the destination node
                    if dest not in pop_state:
                        pop_state[dest] = {'pop': 0, 'delay': speed_mod}
                    
                    pop_state[dest]['pop'] += count
                    
                    if verbose: print(f"  > [Time {t:.1f}] {count} from {group_id} arrived at {dest}")
        
        # Check stop_on_target condition after processing all arrivals for this time step
        if stop_on_target and target_time is not None:
            break

        if saved == total_pop:
             if verbose: print(f"\n--- [Time {t:.1f}] All evacuees are safe! ---")
             break 
             
        # --- 2. DEPARTURES ---
        has_movement = False
        
        for loc in process_order:
            # Check if node exists in state and has people
            loc_data = pop_state.get(loc)
            if not loc_data or loc_data['pop'] <= 0:
                continue
            
            people = loc_data['pop']
            speed_mod = loc_data.get('delay', 1.0)
            
            # Find path
            _, route = dijkstra_shortest_path(graph, loc, exits, edge_flow)
            
            if route:
                src, dst = route[0], route[1]
                edge = (src, dst)
                base_time, cap = get_edge_details(graph, src, dst)
                avail = cap - edge_flow.get(edge, 0)
                
                flow = min(people, avail)
                
                if flow > 0:
     
                    travel = base_time * speed_mod
                    eta = t + travel
                    
                    orig_id = loc if loc in groups else f"{loc}"
                    
                    in_transit.append((eta, dst, orig_id, flow, edge, speed_mod))
                    
                    pop_state[loc]['pop'] -= flow
                    edge_flow[edge] = edge_flow.get(edge, 0) + flow

                    if verbose: print(f"  > [Time {t:.1f}] {flow} from {loc} moving to {dst} (Speed Factor: {speed_mod}x, ETA: {eta:.1f})")

        if not has_movement and not in_transit:
            if verbose: print(f"\n--- [Time {t:.1f}] No further movement possible. ---")
            break 
            
        t += 0.5 
    
    remaining = sum(g['pop'] for g in pop_state.values() if isinstance(g, dict))

    return {
        "Total_Evacuees_Saved": saved,
        "Evacuation_Time_T": t,
        "Remaining_At_Start": remaining,
        "Target_Reached_Time": target_time
    }

def print_scenario_report(name, results, total_pop):
    print("\n" + "-"*50)
    print(f"SCENARIO REPORT: {name}")
    print("-"*50)
    
    saved = results['Total_Evacuees_Saved']
    time_taken = results['Evacuation_Time_T']
    target_time = results.get('Target_Reached_Time')
    
    print(f"> Total Evacuees: {total_pop}")
    print(f"> Evacuees Saved: {saved} ({saved/total_pop*100:.1f}%)")
    print(f"> Evacuees Left:  {results['Remaining_At_Start']}")
    print(f"> Time Elapsed:   {time_taken:.1f} minutes")
    
    if target_time:
         print(f"> Target Reached: {target_time:.1f} minutes")
         
    if time_taken > 0:
        rate = saved / time_taken
        print(f"> Avg Evac Rate:  {rate:.1f} people/min")
        
    print("-"*50 + "\n")

if __name__ == "__main__":
    
    total_pop = sum(group['pop'] for group in starting_groups.values())

    # --- SCENARIO 1 ---
    # print("\n" + "="*50)
    # print("SCENARIO A: QUICK RESPONSE (Save 50 People)")
    # print("="*50)
    res_a = EvacuationOptimization(building_graph, starting_groups, exit_points, target_evacuees=50, stop_on_target=True, verbose=False)
    print_scenario_report("Scenario A (Quick Response)", res_a, total_pop)
    
    # --- SCENARIO 2 ---
    # print("\n" + "="*50)
    # print("SCENARIO B: FIRE EMERGENCY (15 min Limit)")
    # print("="*50)
    res_b = EvacuationOptimization(building_graph, starting_groups, exit_points, time_limit=15.0, verbose=False)
    print_scenario_report("Scenario B (Fire Emergency)", res_b, total_pop)

    # --- SCENARIO 3 ---
    # print("\n" + "="*50)
    # print("SCENARIO C: COMPLETE EVACUATION (Benchmark)")
    # print("="*50)
    res_c = EvacuationOptimization(building_graph, starting_groups, exit_points, verbose=False)
    print_scenario_report("Scenario C (Complete Evacuation)", res_c, total_pop)
