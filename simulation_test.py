import heapq
import copy
import math
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
    'H_F1': [('R101_F1', 0.5, 10), ('Stairs_F1_F2', 1.0, 15),
             ('Exit_A', 1.0, 10), ('Exit_B', 1.0, 10), ('Exit_C', 1.0, 10)],

    'Exit_A': [],
    'Exit_B': [],
    'Exit_C': []
}

starting_groups = {
    'R301_F3': {'pop': 100, 'delay': 1.0},
    'R201_F2': {'pop': 100, 'delay': 4.0},
    'R101_F1': {'pop': 100, 'delay': 1.0}
}

exit_points = ['Exit_A', 'Exit_B', 'Exit_C']


def get_edge_details(graph, u, v):
    for neighbor, time_weight, capacity in graph.get(u, []):
        if neighbor == v:
            return time_weight, capacity
    return None, None


def floyd_warshall_all_pairs(graph):
    nodes = set(graph.keys())
    for u, nbrs in graph.items():
        for v, t, c in nbrs:
            nodes.add(v)
    nodes = list(nodes)

    INF = float('inf')
    dist = {u: {v: INF for v in nodes} for u in nodes}
    nxt = {u: {v: None for v in nodes} for u in nodes}

    for u in nodes:
        dist[u][u] = 0.0
        nxt[u][u] = u

    for u, nbrs in graph.items():
        for v, t, c in nbrs:
            if t < dist[u][v]:
                dist[u][v] = t
                nxt[u][v] = v

    for k in nodes:
        for i in nodes:
            dik = dist[i][k]
            if dik == INF:
                continue
            for j in nodes:
                if dist[k][j] == INF:
                    continue
                cand = dik + dist[k][j]
                if cand < dist[i][j]:
                    dist[i][j] = cand
                    nxt[i][j] = nxt[i][k]

    return dist, nxt


def get_fw_route(start, exits, dist, nxt):
    best_exit = None
    best_d = math.inf

    if start not in dist:
        return math.inf, []

    for ex in exits:
        if ex in dist[start] and dist[start][ex] < best_d:
            best_d = dist[start][ex]
            best_exit = ex

    if best_exit is None or best_d == math.inf:
        return math.inf, []

    route = [start]
    u = start
    while u != best_exit:
        u = nxt[u][best_exit]
        if u is None:
            return math.inf, []
        route.append(u)

    return best_d, route


def dijkstra_shortest_path(graph, start, exits, edge_usage):
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        for n, _, _ in neighbors:
            all_nodes.add(n)
    for n in exits:
        all_nodes.add(n)

    dist = {node: float('inf') for node in all_nodes}
    dist[start] = 0.0
    prev = {}
    pq = [(0.0, start)]

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

            penalty = edge_usage.get(edge, 0)
            d = curr_dist + travel_time + penalty

            if d < dist[neighbor]:
                dist[neighbor] = d
                prev[neighbor] = node
                heapq.heappush(pq, (d, neighbor))

    return float('inf'), []


def build_nx_graph(graph):
    G = nx.DiGraph()
    for u, nbrs in graph.items():
        G.add_node(u)
        for v, t, c in nbrs:
            G.add_edge(u, v, time=t, cap=c)

    pos = {
        'R301_F3': (0, 3), 'H_F3': (2, 3),
        'Stairs_F2_F3': (4, 2.3),

        'R201_F2': (0, 2), 'H_F2': (2, 2),
        'Stairs_F1_F2': (4, 1.3),

        'R101_F1': (0, 1), 'H_F1': (2, 1),

        'Exit_A': (4, 0.3), 'Exit_B': (5, 0.6), 'Exit_C': (6, 0.9)
    }

    for n in G.nodes():
        if n not in pos:
            pos[n] = (0, 0)

    return G, pos


def draw_state(G, pos, pop_state, edge_flow, t, step,
               scenario_name="", edges_highlight=None,
               step_by_step=False):
    plt.clf()
    ax = plt.gca()
    ax.set_title(f"{scenario_name}\nt = {t:.1f}   step = {step}")

    node_sizes = []
    node_colors = []
    labels = {}

    for node in G.nodes():
        pop = 0
        if node in pop_state and isinstance(pop_state[node], dict):
            pop = pop_state[node].get('pop', 0)

        size = 300 + pop * 2
        node_sizes.append(size)

        if node.startswith("Exit"):
            color = "green"
        elif node.startswith("R"):
            color = "orange"
        elif "Stairs" in node:
            color = "red"
        else:
            color = "skyblue"
        node_colors.append(color)

        label = node
        if pop > 0:
            label = f"{node}\n({pop})"
        labels[node] = label

    widths = []
    edge_colors = []
    edges_highlight = edges_highlight or set()

    for u, v in G.edges():
        flow = edge_flow.get((u, v), 0)
        w = 1.0 + flow / 5.0
        widths.append(w)

        if (u, v) in edges_highlight:
            edge_colors.append("red")
        else:
            edge_colors.append("gray")

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths,
                           edge_color=edge_colors,
                           arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=8, ax=ax)

    ax.set_axis_off()
    plt.tight_layout()
    plt.draw()

    if step_by_step:
        print("Press Enter for next step (or Ctrl+C to stop)...")
        input()
    else:
        plt.pause(0.3)


def EvacuationOptimization(graph, groups, exits,
                           time_limit=None,
                           target_evacuees=None,
                           stop_on_target=False,
                           verbose=False,
                           routing_mode='dijkstra',
                           visualize=False,
                           nx_graph=None,
                           pos=None,
                           scenario_name="",
                           step_by_step=False):
    t = 0.0
    saved = 0
    target_time = None

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

    fw_dist = fw_next = None
    if routing_mode == 'floyd':
        fw_dist, fw_next = floyd_warshall_all_pairs(graph)

    if visualize and (nx_graph is None or pos is None):
        nx_graph, pos = build_nx_graph(graph)

    if verbose:
        print(f"--- Starting Simulation ---")
        if time_limit:
            print(f"Time limit: {time_limit} minutes")
        if target_evacuees:
            print(f"Target evacuees: {target_evacuees}")
        print(f"Total evacuees: {total_pop}")
        print(f"Routing mode: {routing_mode}")

    step = 0

    if visualize:
        draw_state(nx_graph, pos, pop_state, edge_flow,
                   t, step, scenario_name, set(), step_by_step)

    for _ in range(2000):
        if time_limit and t > time_limit:
            if verbose:
                print(f"\nTime limit reached at t = {t:.1f}")
            break

        edges_used_step = set()

        for arrival in list(in_transit):
            eta, dest, group_id, count, edge_used, speed_mod = arrival

            if t >= eta:
                in_transit.remove(arrival)
                edge_flow[edge_used] = edge_flow.get(edge_used, 0) - count

                if dest in exits:
                    saved += count
                    if verbose:
                        print(f"[t={t:.1f}] {count} from {group_id} arrived at {dest}")

                    if target_evacuees and saved >= target_evacuees:
                        if target_time is None:
                            target_time = t
                            if verbose:
                                print(f"Target reached: {saved} evacuees at t={t:.1f}")
                        if stop_on_target:
                            if verbose:
                                print("Stopping after reaching target.")
                            if visualize:
                                step += 1
                                draw_state(nx_graph, pos, pop_state,
                                           edge_flow, t, step,
                                           scenario_name, edges_used_step,
                                           step_by_step)
                            remaining = sum(
                                g['pop'] for g in pop_state.values()
                                if isinstance(g, dict)
                            )
                            return {
                                "Total_Evacuees_Saved": saved,
                                "Evacuation_Time_T": t,
                                "Remaining_At_Start": remaining,
                                "Target_Reached_Time": target_time
                            }
                else:
                    if dest not in pop_state:
                        pop_state[dest] = {'pop': 0, 'delay': speed_mod}
                    pop_state[dest]['pop'] += count
                    if verbose:
                        print(f"[t={t:.1f}] {count} from {group_id} arrived at {dest}")

        if saved == total_pop:
            if verbose:
                print(f"\nAll evacuees safe at t={t:.1f}")
            if visualize:
                step += 1
                draw_state(nx_graph, pos, pop_state,
                           edge_flow, t, step,
                           scenario_name, edges_used_step,
                           step_by_step)
            break

        has_movement = False

        for loc in process_order:
            loc_data = pop_state.get(loc)
            if not loc_data or loc_data['pop'] <= 0:
                continue

            people = loc_data['pop']
            speed_mod = loc_data.get('delay', 1.0)

            if routing_mode == 'floyd':
                _, route = get_fw_route(loc, exits, fw_dist, fw_next)
            else:
                _, route = dijkstra_shortest_path(graph, loc, exits, edge_flow)

            if not route or len(route) < 2:
                continue

            src, dst = route[0], route[1]
            base_time, cap = get_edge_details(graph, src, dst)
            if base_time is None or cap is None:
                continue

            edge = (src, dst)
            avail = cap - edge_flow.get(edge, 0)
            if avail <= 0:
                continue

            flow = min(people, avail)

            if flow > 0:
                travel = base_time * speed_mod
                eta = t + travel
                orig_id = loc if loc in groups else f"{loc}"

                in_transit.append((eta, dst, orig_id, flow, edge, speed_mod))
                pop_state[loc]['pop'] -= flow
                edge_flow[edge] = edge_flow.get(edge, 0) + flow
                edges_used_step.add(edge)
                has_movement = True

                if verbose:
                    print(f"[t={t:.1f}] {flow} from {loc} -> {dst} "
                          f"(ETA={eta:.1f}, speed={speed_mod})")

        step += 1
        if visualize:
            draw_state(nx_graph, pos, pop_state,
                       edge_flow, t, step,
                       scenario_name, edges_used_step,
                       step_by_step)

        if not has_movement and not in_transit:
            if verbose:
                print(f"\nNo further movement at t={t:.1f}")
            break

        t += 0.5

    remaining = sum(
        g['pop'] for g in pop_state.values()
        if isinstance(g, dict)
    )

    return {
        "Total_Evacuees_Saved": saved,
        "Evacuation_Time_T": t,
        "Remaining_At_Start": remaining,
        "Target_Reached_Time": target_time
    }


def print_scenario_report(name, results, total_pop):
    print("\n" + "-" * 50)
    print(f"SCENARIO REPORT: {name}")
    print("-" * 50)

    saved = results['Total_Evacuees_Saved']
    time_taken = results['Evacuation_Time_T']
    target_time = results.get('Target_Reached_Time')

    print(f"> Total Evacuees: {total_pop}")
    print(f"> Evacuees Saved: {saved} ({saved / total_pop * 100:.1f}%)")
    print(f"> Evacuees Left:  {results['Remaining_At_Start']}")
    print(f"> Time Elapsed:   {time_taken:.1f} minutes")

    if target_time is not None:
        print(f"> Target Reached: {target_time:.1f} minutes")

    if time_taken > 0:
        rate = saved / time_taken
        print(f"> Avg Evac Rate:  {rate:.1f} people/min")

    print("-" * 50 + "\n")


if __name__ == "__main__":
    total_pop = sum(group['pop'] for group in starting_groups.values())
    G_nx, pos = build_nx_graph(building_graph)

    plt.ion()

    res_a = EvacuationOptimization(
        building_graph,
        starting_groups,
        exit_points,
        target_evacuees=50,
        stop_on_target=True,
        verbose=False,
        routing_mode='dijkstra',
        visualize=True,
        nx_graph=G_nx,
        pos=pos,
        scenario_name="Scenario A – Dijkstra (Quick Response)",
        step_by_step=True          # set False for auto-play
    )
    print_scenario_report("Scenario A – Dijkstra (Quick Response)", res_a, total_pop)

    res_b = EvacuationOptimization(
        building_graph,
        starting_groups,
        exit_points,
        time_limit=15.0,
        verbose=False,
        routing_mode='floyd',
        visualize=False,
        nx_graph=G_nx,
        pos=pos,
        scenario_name="Scenario B – Floyd–Warshall (15-min Limit)",
        step_by_step=False
    )
    print_scenario_report("Scenario B – Floyd–Warshall (Fire Emergency)", res_b, total_pop)

    res_c = EvacuationOptimization(
        building_graph,
        starting_groups,
        exit_points,
        verbose=False,
        routing_mode='floyd',
        visualize=False,
        nx_graph=G_nx,
        pos=pos,
        scenario_name="Scenario C – Floyd–Warshall (Complete Evacuation)",
        step_by_step=False
    )
    print_scenario_report("Scenario C – Floyd–Warshall (Complete Evacuation)", res_c, total_pop)

    plt.ioff()
    plt.show()
