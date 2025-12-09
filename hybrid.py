import heapq
import copy
import time
import math

# --------------------------- Graph + initial data -----------------------------

building_graph = {
    'R301_F3': [('H_F3', 0.5, 5)],
    'H_F3': [('R301_F3', 0.5, 5), ('Stairs_F2_F3', 1.0, 5)],
    'Stairs_F2_F3': [('H_F3', 1.0, 5), ('H_F2', 1.0, 5)],

    'R201_F2': [('H_F2', 0.5, 10)],
    'H_F2': [('R201_F2', 0.5, 10), ('Stairs_F2_F3', 1.0, 5), ('Stairs_F1_F2', 1.0, 15)],
    'Stairs_F1_F2': [('H_F2', 1.0, 15), ('H_F1', 1.0, 15)],

    'R101_F1': [('H_F1', 0.5, 10)],
    'H_F1': [
        ('R101_F1', 0.5, 10),
        ('Stairs_F1_F2', 1.0, 15),
        ('Exit_A', 1.0, 10),
        ('Exit_B', 1.0, 10),
        ('Exit_C', 1.0, 10)
    ],

    'Exit_A': [],
    'Exit_B': [],
    'Exit_C': []
}

starting_groups = {
    'R301_F3': {'pop': 500, 'delay': 1.0},
    'R201_F2': {'pop': 500, 'delay': 4.0},
    'R101_F1': {'pop': 500, 'delay': 1.0}
}

exit_points = ['Exit_A', 'Exit_B', 'Exit_C']


def get_edge_details(graph, u, v):
    """Return (travel_time, capacity) for directed edge u -> v."""
    for neighbor, time_weight, capacity in graph.get(u, []):
        if neighbor == v:
            return time_weight, capacity
    return None, None


# -------- Floyd–Warshall (dynamic programming all-pairs shortest paths) ------

def floyd_warshall_all_pairs(graph):
    """
    Compute all-pairs shortest travel times (ignoring congestion).
    Returns:
        dist[u][v] = shortest travel time from u to v
        nxt[u][v]  = next hop on a shortest path from u to v
    """
    nodes = set(graph.keys())
    for u, nbrs in graph.items():
        for v, t, c in nbrs:
            nodes.add(v)
    nodes = list(nodes)

    INF = float('inf')
    dist = {u: {v: INF for v in nodes} for u in nodes}
    nxt = {u: {v: None for v in nodes} for u in nodes}

    # distance to self = 0
    for u in nodes:
        dist[u][u] = 0.0
        nxt[u][u] = u

    # direct edges
    for u, nbrs in graph.items():
        for v, t, c in nbrs:
            if t < dist[u][v]:
                dist[u][v] = t
                nxt[u][v] = v

    # dynamic programming: allow intermediate node k
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
    """Get best route from 'start' to any exit using FW tables."""
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


# -------------------- Dijkstra (greedy, congestion-aware) ---------------------

def dijkstra_shortest_path(graph, start, exits, edge_usage):
    """
    Dijkstra that includes a congestion penalty based on current edge_usage.
    edge_usage[(u,v)] = number of people currently on that edge.
    Returns (cost, path).
    """
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
            # reconstruct path
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

            # if already at or above capacity, avoid this edge
            if edge_usage.get(edge, 0) >= max_cap:
                continue

            # simple congestion penalty = current flow on that edge
            penalty = edge_usage.get(edge, 0)
            d = curr_dist + travel_time + penalty

            if d < dist[neighbor]:
                dist[neighbor] = d
                prev[neighbor] = node
                heapq.heappush(pq, (d, neighbor))

    return float('inf'), []


# ------------------------ Logging helpers -------------------------------------

def log(msg, verbose, log_lines):
    """
    High-level log:
    - printed to console if verbose
    - ALWAYS appended to log_lines (goes into report)
    """
    if verbose:
        print(msg)
    log_lines.append(msg)


def detail_log(msg, verbose):
    """
    Detailed event log (per-move, arrivals, etc.):
    - printed to console if verbose
    - NOT stored in log_lines (kept out of report)
    """
    if verbose:
        print(msg)


# ------------------------ Simulation display helper --------------------------

def format_step_state(step, t, pop_state, in_transit, saved, total_pop, exits):
    """Return a multi-line string for a per-step snapshot."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append(f"STEP {step:03d}  |  SIM TIME = {t:.1f} minutes")
    lines.append("=" * 70)

    # People at each location
    lines.append("People at nodes:")
    for loc in sorted(pop_state.keys()):
        data = pop_state[loc]
        p = data.get('pop', 0) if isinstance(data, dict) else data
        mark = " [EXIT]" if loc in exits else ""
        lines.append(f"  - {loc:10s}: {p:3d} people{mark}")

    # People in transit
    if in_transit:
        lines.append("\nPeople in transit:")
        for (eta, dest, group_id, count, edge_used, speed_mod, route, alg) in in_transit:
            src, dst = edge_used
            lines.append(
                f"  - {count:3d} from {group_id} going {src} -> {dst} "
                f"(ETA={eta:.1f}, speed={speed_mod:.1f}x, alg={alg})"
            )
            lines.append(f"      full route to exit: {route}")
    else:
        lines.append("\nPeople in transit: none")

    # Saved so far
    pct = (saved / total_pop * 100.0) if total_pop > 0 else 0.0
    lines.append(f"\nSaved so far: {saved}/{total_pop} ({pct:.1f}%)")
    lines.append("=" * 70)
    return "\n".join(lines)


# ------------------------ Evacuation simulation core -------------------------

def EvacuationOptimization(
    graph,
    groups,
    exits,
    time_limit=None,
    target_evacuees=None,
    stop_on_target=False,
    verbose=False,
    routing_mode='dijkstra',
    show_steps=False
):
    """
    Simulate evacuation using one of:
      routing_mode='dijkstra' – greedy, congestion-aware
      routing_mode='floyd'    – global DP shortest paths (no congestion in route choice)
      routing_mode='hybrid'   – compute both (Dijkstra + Floyd) and pick per node

    If show_steps=True, store a full "snapshot" every half-step in the log.
    If verbose=True, also print detailed per-move messages to the console.
    """
    t = 0.0
    saved = 0
    target_time = None

    total_pop = sum(group['pop'] for group in groups.values())
    pop_state = copy.deepcopy(groups)   # node -> {'pop', 'delay'}

    # in_transit: (eta, dest, group_id, count, edge_used, speed_mod, route, alg_used)
    in_transit = []
    edge_flow = {}       # (u,v) -> current number of evacuees on that edge

    process_order = [
        'R301_F3', 'H_F3',
        'Stairs_F2_F3', 'R201_F2',
        'H_F2', 'Stairs_F1_F2',
        'R101_F1', 'H_F1'
    ]

    # Precompute FW tables once if needed
    fw_dist = fw_next = None
    if routing_mode in ('floyd', 'hybrid'):
        fw_dist, fw_next = floyd_warshall_all_pairs(graph)

    log_lines = []

    # Header
    log(f"--- Starting Simulation (routing = {routing_mode}) ---", verbose, log_lines)
    if time_limit is not None:
        log(f"Time limit: {time_limit} minutes", verbose, log_lines)
    if target_evacuees is not None:
        log(f"Target evacuees: {target_evacuees}", verbose, log_lines)
    log(f"Total evacuees in building: {total_pop}\n", verbose, log_lines)

    step = 0
    if show_steps:
        snapshot = format_step_state(step, t, pop_state, in_transit,
                                     saved, total_pop, exits)
        log(snapshot, verbose, log_lines)

    # Up to 2000 half-steps = 1000 minutes max
    for _ in range(2000):

        # stop if building is unsafe
        if time_limit is not None and t > time_limit:
            log(f"[t={t:.1f}] Time limit reached.", verbose, log_lines)
            break

        # ------------------ 1. ARRIVALS ------------------
        for arrival in list(in_transit):
            (eta, dest, group_id, count,
             edge_used, speed_mod, route, alg_used) = arrival

            if t >= eta:
                in_transit.remove(arrival)
                edge_flow[edge_used] = edge_flow.get(edge_used, 0) - count

                if dest in exits:
                    saved += count
                    detail_log(
                        f"[t={t:.1f}] {count} from {group_id} arrived at {dest} (via {alg_used}).",
                        verbose
                    )

                    if target_evacuees is not None and saved >= target_evacuees:
                        if target_time is None:
                            target_time = t
                            log(f"[t={t:.1f}] Target reached with {saved} evacuees.",
                                verbose, log_lines)
                        if stop_on_target:
                            remaining = sum(
                                g['pop'] for g in pop_state.values()
                                if isinstance(g, dict)
                            )
                            log(f"[t={t:.1f}] Stopping after reaching target.\n",
                                verbose, log_lines)
                            if show_steps:
                                step += 1
                                snapshot = format_step_state(step, t, pop_state,
                                                             in_transit, saved,
                                                             total_pop, exits)
                                log(snapshot, verbose, log_lines)
                            return {
                                "Total_Evacuees_Saved": saved,
                                "Evacuation_Time_T": t,
                                "Remaining_At_Start": remaining,
                                "Target_Reached_Time": target_time,
                                "Log": log_lines
                            }
                else:
                    # people arrive into intermediate nodes (hallways, stairs)
                    if dest not in pop_state:
                        pop_state[dest] = {'pop': 0, 'delay': speed_mod}
                    pop_state[dest]['pop'] += count
                    detail_log(
                        f"[t={t:.1f}] {count} from {group_id} arrived at {dest}.",
                        verbose
                    )

        # everyone safe
        if saved == total_pop:
            log(f"[t={t:.1f}] All evacuees are safe.\n", verbose, log_lines)
            if show_steps:
                step += 1
                snapshot = format_step_state(step, t, pop_state, in_transit,
                                             saved, total_pop, exits)
                log(snapshot, verbose, log_lines)
            break

        # ------------------ 2. DEPARTURES ------------------
        has_movement = False

        for loc in process_order:
            loc_data = pop_state.get(loc)
            if not loc_data or loc_data['pop'] <= 0:
                continue

            people = loc_data['pop']
            speed_mod = loc_data.get('delay', 1.0)

            # choose path depending on routing_mode
            if routing_mode == 'floyd':
                d_cost, fw_route = get_fw_route(loc, exits, fw_dist, fw_next)
                route = fw_route
                alg_used = "Floyd–Warshall"

            elif routing_mode == 'dijkstra':
                d_cost, d_route = dijkstra_shortest_path(graph, loc, exits, edge_flow)
                route = d_route
                alg_used = "Dijkstra"

            elif routing_mode == 'hybrid':
                # 1) dynamic, congestion-aware route
                dij_cost, dij_route = dijkstra_shortest_path(graph, loc, exits, edge_flow)
                # 2) global static route
                fw_cost, fw_route = get_fw_route(loc, exits, fw_dist, fw_next)

                # Decide which to use.
                # Rule: if Dijkstra finds a path and has <= cost than FW (or FW fails), use Dijkstra.
                # Otherwise use FW.
                if (not dij_route) and (not fw_route):
                    route = []
                    d_cost = math.inf
                    alg_used = "Hybrid (no route)"
                elif (not fw_route):
                    route = dij_route
                    d_cost = dij_cost
                    alg_used = "Hybrid (Dijkstra only)"
                elif (not dij_route):
                    route = fw_route
                    d_cost = fw_cost
                    alg_used = "Hybrid (Floyd–Warshall only)"
                else:
                    if dij_cost <= fw_cost:
                        route = dij_route
                        d_cost = dij_cost
                        alg_used = "Hybrid (Dijkstra preferred)"
                    else:
                        route = fw_route
                        d_cost = fw_cost
                        alg_used = "Hybrid (Floyd–Warshall preferred)"
            else:
                # fallback: treat anything else as dijkstra
                d_cost, d_route = dijkstra_shortest_path(graph, loc, exits, edge_flow)
                route = d_route
                alg_used = "Dijkstra"

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

            # how many can we send through this edge now?
            flow = min(people, avail)

            if flow > 0:
                travel = base_time * speed_mod
                eta = t + travel
                group_id = loc  # keep origin label simple

                in_transit.append(
                    (eta, dst, group_id, flow, edge, speed_mod, route, alg_used)
                )
                pop_state[loc]['pop'] -= flow
                edge_flow[edge] = edge_flow.get(edge, 0) + flow
                has_movement = True

                detail_log(
                    f"[t={t:.1f}] ASSIGN {flow} from {loc} -> {dst} "
                    f"(alg={alg_used}, travel={base_time:.1f}, "
                    f"speed={speed_mod:.1f}x, ETA={eta:.1f})",
                    verbose
                )
                detail_log(
                    f"        full route to exit: {route}",
                    verbose
                )

        if not has_movement and not in_transit:
            log(f"[t={t:.1f}] No further movement possible.\n", verbose, log_lines)
            if show_steps:
                step += 1
                snapshot = format_step_state(step, t, pop_state, in_transit,
                                             saved, total_pop, exits)
                log(snapshot, verbose, log_lines)
            break

        # ------------------ 3. ADVANCE TIME ------------------
        step += 1
        if show_steps:
            snapshot = format_step_state(step, t, pop_state, in_transit,
                                         saved, total_pop, exits)
            log(snapshot, verbose, log_lines)

        t += 0.5

    remaining = sum(
        g['pop'] for g in pop_state.values()
        if isinstance(g, dict)
    )

    return {
        "Total_Evacuees_Saved": saved,
        "Evacuation_Time_T": t,
        "Remaining_At_Start": remaining,
        "Target_Reached_Time": target_time,
        "Log": log_lines
    }


def print_scenario_report(name, results, total_pop, cpu_time_ms=None):
    print("\n" + "-" * 50)
    print(f"SCENARIO REPORT: {name}")
    print("-" * 50)

    saved = results['Total_Evacuees_Saved']
    time_taken = results['Evacuation_Time_T']
    target_time = results.get('Target_Reached_Time')

    print(f"> Total Evacuees in Building: {total_pop}")
    print(f"> Evacuees Saved:            {saved} ({saved / total_pop * 100:.1f}%)")
    print(f"> Evacuees Left Inside:      {results['Remaining_At_Start']}")
    print(f"> Simulation Time Elapsed:   {time_taken:.1f} minutes")

    if target_time is not None:
        print(f"> Time When Target Reached:  {target_time:.1f} minutes")

    if time_taken > 0:
        rate = saved / time_taken
        print(f"> Avg Evacuation Rate:       {rate:.1f} people/min")

    if cpu_time_ms is not None:
        print(f"> CPU Time (algorithm):      {cpu_time_ms:.4f} ms")

    print("-" * 50 + "\n")


def append_report_to_file(filename, scenario_name, results, total_pop, cpu_time_ms):
    """Append a report (summary + clean step-by-step log + CPU time) to a text file."""
    with open(filename, "a", encoding="utf-8") as f:
        # Scenario header
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"SCENARIO REPORT: {scenario_name}\n")
        f.write("=" * 80 + "\n\n")

        saved = results['Total_Evacuees_Saved']
        time_taken = results['Evacuation_Time_T']
        target_time = results.get('Target_Reached_Time')

        # --- Summary block ---
        f.write("Summary\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total evacuees in building: {total_pop}\n")
        f.write(f"Evacuees saved:             {saved} ({saved / total_pop * 100:.1f}%)\n")
        f.write(f"Evacuees left inside:       {results['Remaining_At_Start']}\n")
        f.write(f"Simulation time elapsed:    {time_taken:.1f} minutes\n")
        if target_time is not None:
            f.write(f"Time when target reached:   {target_time:.1f} minutes\n")
        if time_taken > 0:
            rate = saved / time_taken
            f.write(f"Average evacuation rate:    {rate:.1f} people/min\n")

        # CPU time (implementation overhead comparison)
        f.write(f"CPU time for algorithm:     {cpu_time_ms:.4f} ms\n")

        # --- Clean step-by-step log block ---
        f.write("\nStep-by-step simulation log\n")
        f.write("-" * 80 + "\n")

        for entry in results.get("Log", []):
            f.write(entry + "\n")

        f.write("\n")


# ---------------------------------- main -------------------------------------

if __name__ == "__main__":
    total_pop = sum(group['pop'] for group in starting_groups.values())
    report_file = "evacuation_report.txt"
    time_limit = 15.0

    # Reset report file
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Evacuation Simulation Report\n")
        f.write("========================================\n\n")

    # =============== DIJKSTRA (time-limit) ===============
    print("\n================= DIJKSTRA (time-limit) =================")
    start_d = time.perf_counter()
    res_d = EvacuationOptimization(
        building_graph,
        starting_groups,
        exit_points,
        time_limit=time_limit,
        target_evacuees=None,
        stop_on_target=False,
        verbose=False,
        routing_mode='dijkstra',
        show_steps=False
    )
    end_d = time.perf_counter()
    cpu_d = (end_d - start_d) * 1000.0

    print_scenario_report("Time-Limit Scenario (Dijkstra)", res_d, total_pop, cpu_d)
    append_report_to_file(
        report_file,
        "Time-Limit Scenario (Dijkstra)",
        res_d,
        total_pop,
        cpu_d
    )

    # =============== FLOYD–WARSHALL (time-limit) ===============
    print("\n============= FLOYD–WARSHALL (time-limit) ===============")
    start_f = time.perf_counter()
    res_f = EvacuationOptimization(
        building_graph,
        starting_groups,
        exit_points,
        time_limit=time_limit,
        target_evacuees=None,
        stop_on_target=False,
        verbose=False,
        routing_mode='floyd',
        show_steps=False
    )
    end_f = time.perf_counter()
    cpu_f = (end_f - start_f) * 1000.0

    print_scenario_report("Time-Limit Scenario (Floyd–Warshall)", res_f, total_pop, cpu_f)
    append_report_to_file(
        report_file,
        "Time-Limit Scenario (Floyd–Warshall)",
        res_f,
        total_pop,
        cpu_f
    )

    # =============== HYBRID (Dijkstra + Floyd) ===============
    print("\n================= HYBRID (Dijkstra + Floyd) =================")
    start_h = time.perf_counter()
    res_h = EvacuationOptimization(
        building_graph,
        starting_groups,
        exit_points,
        time_limit=time_limit,
        target_evacuees=None,
        stop_on_target=False,
        verbose=False,
        routing_mode='hybrid',
        show_steps=False
    )
    end_h = time.perf_counter()
    cpu_h = (end_h - start_h) * 1000.0

    print_scenario_report("Time-Limit Scenario (Hybrid)", res_h, total_pop, cpu_h)
    append_report_to_file(
        report_file,
        "Time-Limit Scenario (Hybrid)",
        res_h,
        total_pop,
        cpu_h
    )
