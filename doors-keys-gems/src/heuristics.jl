using PDDL, SymbolicPlanners
using Graphs, SimpleWeightedGraphs
using IterTools

import SymbolicPlanners:
    compute, precompute!, is_precomputed, filter_available, ReusableTreeGoal

include("utils.jl")

## GoalManhattan ##

"""
    GoalManhattan

Custom relaxed distance heuristic to goal objects. Estimates the cost of 
collecting all goal objects by computing the distance between all goal objects
and the agent, then returning the minimum distance plus the number of remaining
goals to satisfy.
"""
mutable struct GoalManhattan <: Heuristic
    agents::Vector{Const}
end

GoalManhattan() = GoalManhattan(Const[])
GoalManhattan(domain::Domain, state::State) =
    GoalManhattan(PDDL.get_objects(domain, state, :agent))

is_precomputed(h::GoalManhattan) = !isempty(h.agents)

function precompute!(h::GoalManhattan,
                     domain::Domain, state::State, spec::Specification)
    h.agents = PDDL.get_objects(domain, state, :agent)
    return h
end

function compute(h::GoalManhattan,
                 domain::Domain, state::State, spec::Specification)
    # Decompose goals into DNF clauses, then flatten into single clause
    goals = PDDL.flatten_conjs(decompose_goals(domain, state, spec))
    n_agents = length(h.agents)
    # Look up movement and wait costs for each agent
    move_costs, wait_costs, _, _ = extract_action_costs(spec, h.agents)
    dists = map(goals) do goal
        if state[goal] return 0.0 end
        # Compute distance from focal agent to goal item
        agent, item = goal.args
        agent_idx = findfirst(==(agent), h.agents)
        item_loc = get_obj_loc(state, item; check_has=true)
        agent_loc = get_obj_loc(state, agent)
        agent_item_dist = sum(abs.(agent_loc .- item_loc))
        # Compute minimum distance for other agents to pick up and pass item
        min_other_dist = Inf
        for (other_idx, other) in enumerate(h.agents)
            # Skip if object is not an item
            PDDL.get_objtype(state, item) == :door && continue
            # Skip focal agent
            other == agent && continue 
            # Skip if other agent cannot pick up item
            state[Compound(:forbidden, Term[other, item])] && continue
            # Add distance from item to focal agent
            other_dist = agent_item_dist * minimum(move_costs)
            # Add distance from other agent to item
            if !state[Compound(:has, Term[other, item])]
                other_loc = get_obj_loc(state, other)
                other_item_dist = sum(abs.(other_loc .- item_loc))
                other_dist += other_item_dist * move_costs[other_idx]
                # Add costs of other agents' wait actions
                for idx in 1:n_agents
                    idx == other_idx && continue
                    other_dist += other_item_dist * wait_costs[idx]
                end
            end
            min_other_dist = min(min_other_dist, other_dist)
        end
        # Compute movement cost for focal agent
        agent_dist = agent_item_dist * move_costs[agent_idx]
        # Add costs of other agents' wait actions
        for other_idx in 1:n_agents
            other_idx == agent_idx && continue
            agent_dist += agent_item_dist * wait_costs[other_idx]
        end
        return min(agent_dist, min_other_dist)
    end
    # Compute minimum distance to any goal
    min_dist = length(dists) > 0 ? minimum(dists) : 0.0
    return min_dist
end

## RelaxedMazeDist ##

"""
    RelaxedMazeDist([planner::Planner])

Custom relaxed distance heuristic. Estimates the cost of achieving the goal 
by removing all doors from the state, then computing the length of the plan 
to achieve the goal in the relaxed state.

A `planner` can specified to compute the relaxed plan. By default this is
`AStarPlanner(heuristic=GoalManhattan())`.
"""
function RelaxedMazeDist()
    planner = AStarPlanner(GoalManhattan())
    return RelaxedMazeDist(planner)
end

function RelaxedMazeDist(heuristic::Heuristic)
    planner = AStarPlanner(heuristic)
    return RelaxedMazeDist(planner)
end

function RelaxedMazeDist(planner::Planner)
    heuristic = PlannerHeuristic(planner, s_transform=unlock_doors)
    heuristic = memoized(heuristic)
    return heuristic
end

"Unlocks all doors in the state."
function unlock_doors(state::State)
    state = copy(state)
    for d in PDDL.get_objects(state, :door)
        state[Compound(:locked, Term[d])] = false
    end
    return state
end


## DoorsKeysMSTHeuristic ##

"""
    DoorsKeysMSTHeuristic(
        dist_method = :min_dist,
        max_recurse_iters = 5
    )

A heuristic for PDDL gridworld domains that estimates the cost of achieving a
set of goals by computing the minimum spanning tree (MST) of the graph induced
by the locations of agents, goal objects, and the required keys and doors.

The MST is computed for every possible set of required keys and doors, and 
the minimum cost is returned. Required keys and doors are computed by finding
traversing the room connectivity graph to the goal, then recursively finding
the objects needed to reach newly discovered keys, similar to [1].

[1] D. Aversa, S. Sardina, and S. Vassos, “Pruning and preprocessing methods for
inventory-aware pathfinding,” in 2016 IEEE Conference on Computational
Intelligence and Games (CIG), Sep. 2016, pp. 1–8. doi: 10.1109/CIG.2016.7860417.
"""
mutable struct DoorsKeysMSTHeuristic <: Heuristic
    dist_method::Symbol # Object distance estimation method
    max_recurse_iters::Int # Maximum recursion for necessary key/door analysis
    agents::Vector{Const} # Agent objects
    items::Vector{Const} # Item objects
    doors::Vector{Const} # Door objects
    objects::Vector{Const} # All physical objects
    unlockers::Dict{Const, Vector{Const}} # Map from doors to unlocking keys
    room_labels::Matrix{Int} # Room/door labels for each grid cell
    room_graph::SimpleGraph{Int} # Room connectivity graph
    cell_dists::Union{Matrix{Float64}, Nothing} # Distance between grid cells
    kd_sets::Vector{Vector{NTuple{2, Vector{Const}}}} # Necessary keys/doors
    necessary_keys::Vector{Const} # All necessary keys
    necessary_doors::Vector{Const} # All necessary doors
    goal_items::Vector{Const} # Goal items to be collected
    goal_clauses::Vector{Term} # DNF clauses for current goal
    goal_hash::UInt # Hash of current goal
    min_action_costs::NTuple{4, Float64} # Current minimal action costs
    function DoorsKeysMSTHeuristic(dist_method, max_recurse_iters)
        return new(dist_method, max_recurse_iters)
    end
end

function DoorsKeysMSTHeuristic(;
    dist_method::Symbol = :min_dist,
    max_recurse_iters::Int = 5
)
    return DoorsKeysMSTHeuristic(dist_method, max_recurse_iters)
end

is_precomputed(h::DoorsKeysMSTHeuristic) = isdefined(h, :room_labels)

function precompute!(h::DoorsKeysMSTHeuristic,
                     domain::Domain, state::State)
    # Precompute and sort objects
    h.agents = sort!(PDDL.get_objects(domain, state, :agent), by=x->x.name)
    h.items = sort!(PDDL.get_objects(domain, state, :item), by=x->x.name)
    h.doors = sort!(collect(PDDL.get_objects(state, :door)), by=x->x.name)
    h.objects = Const[h.agents; h.items; h.doors]
    # Label rooms and construct room graph
    h.room_labels = label_rooms_and_doors(domain, state; doors=h.doors)
    h.room_graph = construct_room_graph(domain, state, h.room_labels;
                                        doors=h.doors)
    # Compute unlockers 
    h.unlockers = compute_unlockers(domain, state)
    # Compute grid cell distances
    if h.dist_method == :min_dist
        h.cell_dists = compute_cell_distances(state[pddl"(walls)"])
    else
        h.cell_dists = nothing
    end
    return h
end

function precompute!(h::DoorsKeysMSTHeuristic,
                     domain::Domain, state::State, spec::Specification)
    # Precompute non-goal-specific information
    if !isdefined(h, :room_labels) precompute!(h, domain, state) end
    # Check if goal has changed
    goal_hash = spec isa ReusableTreeGoal ? hash(spec.spec) : hash(spec)
    if isdefined(h, :goal_hash) && h.goal_hash == goal_hash return h end
    h.goal_hash = goal_hash
    # Decompose goals into DNF clauses
    h.goal_clauses = convert(Vector{Term}, decompose_goals(domain, state, spec))
    # Compute necessary keys and doors for each clause
    h.goal_items = Const[]
    h.necessary_keys = Const[]
    h.necessary_doors = Const[]
    h.kd_sets = map(h.goal_clauses) do clause
        clause = PDDL.flatten_conjs(clause)
        goal_agents = unique!([term.args[1]::Const for term in clause])
        goal_items = unique!([term.args[2]::Const for term in clause])
        append!(h.goal_items, goal_items)
        unique!(h.goal_items)
        agent = first(goal_agents)
        to_reach = [goal_agents; goal_items]
        kd_sets = find_necessary_keys_and_doors_recursive(
            domain, state, agent, to_reach, h.room_labels, h.room_graph;
            all_doors = h.doors, all_agents = h.agents,
            unlockers = h.unlockers, max_iters = h.max_recurse_iters
        )
        for (keys, doors) in kd_sets
            append!(h.necessary_keys, keys)
            unique!(h.necessary_keys)
            append!(h.necessary_doors, doors)
            unique!(h.necessary_doors)
        end
        return kd_sets
    end
    # Extract minimum action costs
    h.min_action_costs = extract_min_action_costs(spec, h.agents)
    return h
end

function compute(h::DoorsKeysMSTHeuristic,
                 domain::Domain, state::State, spec::Specification)
    # Precompute goal specific information
    precompute!(h, domain, state, spec)
    # Check if there are any goal clauses to satisfy
    if isempty(h.goal_clauses) return 0.0 end
    # Extract action costs
    move_cost, wait_cost, pickup_cost, unlock_cost = h.min_action_costs
    # Construct location graph
    loc_graph  = construct_location_graph(
        domain, state, h.room_labels, h.room_graph;
        agents = h.agents, items = h.items,
        doors = h.doors, objects = h.objects,
        dist_estim = h.dist_method, cell_dists = h.cell_dists,
        move_cost, wait_cost, pickup_cost, unlock_cost
    )
    # Compute MST cost for each goal clause
    loc_objects = [h.objects; h.objects]
    cost = minimum(zip(h.goal_clauses, h.kd_sets)) do (clause, kd_sets)
        clause = PDDL.flatten_conjs(clause)
        satisfy(domain, state, clause) && return 0.0
        goal_agents = unique!([term.args[1]::Const for term in clause])
        goal_items = unique!([term.args[2]::Const for term in clause])
        agent = first(goal_agents)
        to_reach = [goal_agents; goal_items]
        c = mst_estim_cost(
            domain, state, goal_agents, goal_items,
            h.room_labels, h.room_graph, loc_objects, loc_graph;
            all_doors = h.doors, unlockers = h.unlockers,
            max_iters = h.max_recurse_iters
        )
        if any(state[Compound(:active, Term[a])] for a in goal_agents)
            return max(0.0, c - wait_cost)
        else
            return c
        end
    end
    return cost
end

function filter_available(h::DoorsKeysMSTHeuristic,
                          domain::Domain, state::State, spec::Specification)
    # Precompute goal specific information
    precompute!(h, domain, state, spec)
    # Return iterator that filters out actions involving irrelevant keys/doors
    return Iterators.filter(available(domain, state)) do act
        if act.name == :pickup
            item = act.args[2]
            PDDL.get_objtype(state, item) != :key && return true
            item in h.necessary_keys && return true
            item in h.goal_items && return true
            return false
        elseif act.name == :unlock
            key, door = act.args[2], act.args[3]
            door in h.necessary_doors || return false
            key in h.goal_items && return true
            key in h.necessary_keys || return false
            return true
        elseif act.name == :handover
            item = act.args[3]
            PDDL.get_objtype(state, item) != :key && return true
            item in h.necessary_keys && return true
            item in h.goal_items && return true
            return false
        end
        return true
    end
end

"""
    label_connected_components(grid::AbstractMatrix, val = 0)

Labels the connected components of a grid, where `val` specifies the foreground
value. Returns a matrix of the same size as `grid` with the labels.
"""
function label_connected_components(grid::AbstractMatrix, val::Integer = 0)
    cur_label = 1
    labels = zeros(Int, size(grid))
    queue = Vector{CartesianIndex{2}}()
    indices = CartesianIndices(grid)
    dirs = (CartesianIndex(-1, 0), CartesianIndex(1, 0),
            CartesianIndex(0, -1), CartesianIndex(0, 1))
    for ij in indices
        labels[ij] == 0 && grid[ij] == val || continue
        push!(queue, ij)
        while !isempty(queue)
            yx = pop!(queue)
            labels[yx] = cur_label
            for d in dirs
                cell = yx + d
                1 <= cell[1] <= size(grid, 1) || continue 
                1 <= cell[2] <= size(grid, 2) || continue
                labels[cell] == 0 && grid[cell] == val || continue
                push!(queue, cell)
            end
        end
        cur_label += 1
    end
    return labels
end

"""
    label_rooms_and_doors(domain, state)

Labels the rooms (connected regions) and doors in a PDDL gridworld state.
Returns a matrix of the same size as the grid with the labels. Door labels are
assigned to the maximum room label plus the door index.
"""
function label_rooms_and_doors(
    domain::Domain, state::State;
    doors = sort!(collect(PDDL.get_objects(state, :door)), by = x -> x.name)
)
    walls = copy(state[pddl"(walls)"])
    for door in doors
        x, y = get_obj_loc(state, door)
        walls[y, x] = true
    end
    labels = label_connected_components(walls, false)
    n_rooms = maximum(labels)
    for (i, door) in enumerate(doors)
        x, y = get_obj_loc(state, door)
        labels[y, x] = n_rooms + i
    end
    return labels
end

"""
    construct_room_graph(domain, state, room_labels)

Constructs a graph where the nodes are the rooms and doors in a PDDL gridworld
state, and the edges are the connections between them. Each door space 
connects to adjacent rooms.
"""
function construct_room_graph(
    domain::Domain, state::State,
    room_labels::AbstractMatrix;
    doors = sort!(collect(PDDL.get_objects(state, :door)), by = x -> x.name)
)
    n_rooms_and_doors = maximum(room_labels)
    n_rooms = n_rooms_and_doors - length(doors)
    graph = SimpleGraph(n_rooms_and_doors)
    dirs = (CartesianIndex(-1, 0), CartesianIndex(1, 0),
            CartesianIndex(0, -1), CartesianIndex(0, 1))
    height, width = size(room_labels)
    for (i, door) in enumerate(doors)
        x, y = get_obj_loc(state, door)
        loc = CartesianIndex(y, x)
        for d in dirs
            cell = loc + d
            (1 <= cell[1] <= height) && (1 <= cell[2] <= width) || continue 
            room_labels[cell] != 0 || continue
            add_edge!(graph, n_rooms+i, room_labels[cell]) 
        end
    end
    return graph
end

"""
    construct_location_graph(domain, state, room_labels, room_graph)

Constructs a graph where the nodes are the agents, items, and doors in a PDDL
gridworld state, and the edges are the connections between them. Edges 
are weighted by the estimated distance between the locations.
"""
function construct_location_graph(
    domain::Domain, state::State,
    room_labels::AbstractMatrix, room_graph::AbstractGraph;
    agents = sort!(PDDL.get_objects(domain, state, :agent), by = x -> x.name),
    items = sort!(PDDL.get_objects(domain, state, :item), by = x -> x.name),
    doors = sort!(collect(PDDL.get_objects(state, :door)), by = x -> x.name),
    objects = Const[agents; items; doors],
    dist_estim::Symbol = :manhattan,
    cell_dists::Union{AbstractMatrix, Nothing} = nothing,
    move_cost = 1.0, wait_cost = 1.0, pickup_cost = 1.0, unlock_cost = 1.0
)
    n_agents, n_items, n_doors = length(agents), length(items), length(doors)
    n_objects = length(objects)
    n_locations = n_objects * 2
    graph = SimpleWeightedGraph(n_locations)
    height, width = size(room_labels)
    lin_idxs = LinearIndices(room_labels)
    # Add objects and their locations to graph
    for i in 1:n_objects
        is_agent_i = i <= n_agents
        is_item_i = n_agents < i <= n_agents + n_items
        is_door_i = n_agents + n_items < i <= n_objects
        x_i, y_i = get_obj_loc(state, objects[i])
        # Handle off-grid items
        has_i = false
        if x_i < 0 || y_i < 0 
            for agent in agents
                if state[Compound(:has, Term[agent, objects[i]])]
                    x_i, y_i = get_obj_loc(state, agent)
                    has_i = true # Agent is holding item
                    break
                end
            end
            (x_i < 0 || y_i < 0) && continue
        end
        # Add edge from location to associated object
        cost = if is_agent_i # Agents
            0.0
        elseif is_item_i # Items
            has_i ? 0.0 : pickup_cost
        else # Doors
            locked = state[Compound(:locked, Term[objects[i]])]
            locked ? max(0.0, unlock_cost - move_cost) : 0.0
        end
        add_edge!(graph, i, i + n_objects, cost == 0.0 ? eps() : cost)
        # Look-up room label for location
        room_i = room_labels[y_i, x_i]
        nbs_i = neighbors(room_graph, room_i)
        for j in 1:(i-1)
            is_agent_j = j <= n_agents
            is_item_j = n_agents < j <= n_agents + n_items
            is_door_j = n_agents + n_items < j <= n_objects
            x_j, y_j = get_obj_loc(state, objects[j])
            # Handle off-grid items
            has_j = false
            if x_j < 0 || y_j < 0
                for agent in agents
                    if state[Compound(:has, Term[agent, objects[j]])]
                        has_j = true # Agent is holding item
                        x_j, y_j = get_obj_loc(state, agent)
                        break
                    end
                end
                (x_j < 0 || y_j < 0) && continue
            end
            # Check that locations are in the same or adjacent rooms
            room_j = room_labels[y_j, x_j]
            nbs_j = neighbors(room_graph, room_j)
            if !(room_i == room_j || room_j in nbs_i ||
                 is_door_i && is_door_j && !isempty(intersect(nbs_i, nbs_j)))
                continue
            end        
            # Estimate distance between locations
            if dist_estim == :manhattan
                dist = abs(x_i - x_j) + abs(y_i - y_j)
            elseif dist_estim == :min_dist && !isnothing(cell_dists)
                dist = cell_dists[lin_idxs[y_i, x_i], lin_idxs[y_j, x_j]]
            else
                error("Invalid distance estimation method.")
            end
            # Compute cost of movement, accounting for other agents
            cost = if i <= n_agents && j <= n_agents
                dist * move_cost
            else
                dist * move_cost + dist * wait_cost
            end
            # Add edge between locations
            add_edge!(graph, i, j, cost == 0.0 ? eps() : cost)
        end
    end
    return graph
end

"""
    compute_cell_distances(grid, val = 0)

Compute the distance between each unoccupied cell in a grid to every other cell,
where `val` specifies the foreground value. 
"""
function compute_cell_distances(grid::AbstractMatrix, val = 0)
    cell_dists = ones(length(grid), length(grid)) .* Inf
    dirs = (CartesianIndex(-1, 0), CartesianIndex(1, 0),
            CartesianIndex(0, -1), CartesianIndex(0, 1))
    height, width = size(grid)
    lin_idxs = LinearIndices(grid)
    for (i, idx) in enumerate(CartesianIndices(grid))
        grid[idx] == val || continue
        cell_dists[i, i] = 0
        queue = [(idx, 0)]
        while !isempty(queue)
            idx, dist = popfirst!(queue)
            for d in dirs
                cell = idx + d
                (1 <= cell[1] <= height) && (1 <= cell[2] <= width) || continue
                grid[cell] == val || continue
                j = lin_idxs[cell]
                cell_dists[i, j] == Inf || continue
                cell_dists[i, j] = dist + 1
                push!(queue, (cell, dist + 1))
            end
        end
    end
    return cell_dists
end

"""
    compute_unlockers(domain, state)

Computes the keys that unlock each door in a PDDL gridworld state. Returns a
dictionary mapping doors to keys.
"""
function compute_unlockers(
    domain::Domain, state::State; 
    doors = PDDL.get_objects(state, :door),
    keys = PDDL.get_objects(state, :key)
)
    unlockers = Dict{Const, Vector{Const}}()
    for door in doors
        color = get_obj_color(state, door)
        ks = Const[k for k in keys if state[Compound(:iscolor, Term[k, color])]]
        unlockers[door] = ks
    end
    return unlockers
end


"""
    find_all_paths(graph, start, goal)

Finds all paths between two nodes in a graph.
"""
function find_all_paths(graph::AbstractGraph, start::Int, goal::Int)
    all_paths = Vector{Int}[]
    queue = [(start, Int[start])]
    while !isempty(queue)
        u, path = popfirst!(queue)
        if u == goal
            push!(all_paths, path)
            continue
        end
        for v in neighbors(graph, u)
            v in path && continue
            new_path = push!(copy(path), v)
            push!(queue, (v, new_path))
        end
    end
    return all_paths
end

"""
    find_necessary_doors(domain, state, r_start, r_goal, room_graph)
    find_necessary_doors(domain, state, agent, item, room_labels, room_graph)

Finds the necessary doors to traverse between two rooms in a PDDL gridworld
state. The first method takes room indices, and the second takes agent and item
PDDL objects. Returns a vector of door sets, where each door set is a vector of
doors that must be traversed in order to reach the goal room.
"""
function find_necessary_doors(
    domain::Domain, state::State,
    r_start::Int, r_goal::Int, room_graph::AbstractGraph;
    all_doors = sort!(collect(PDDL.get_objects(state, :door)), by=x->x.name)
)
    n_rooms_and_doors = nv(room_graph)
    n_rooms = n_rooms_and_doors - length(all_doors)
    all_paths = find_all_paths(room_graph, r_start, r_goal)
    door_sets = map(all_paths) do path
        door_idxs = filter(>(n_rooms), path) .- n_rooms 
        return all_doors[door_idxs]::Vector{Const}
    end
    return door_sets
end

function find_necessary_doors(
    domain::Domain, state::State,
    agent::Const, item::Const,
    room_labels::AbstractMatrix, room_graph::AbstractGraph;
    all_doors = sort!(collect(PDDL.get_objects(state, :door)), by=x->x.name)
)
    x, y = get_obj_loc(state, agent)
    r_start = room_labels[y, x]
    x, y = get_obj_loc(state, item, check_has=true)
    r_goal = room_labels[y, x]
    return find_necessary_doors(domain, state, r_start, r_goal,
                                room_graph; all_doors)
end

"""
    find_necessary_keys(domain, state, doors, unlockers)
    find_necessary_keys(domain, state, door_sets, unlockers)

Finds the necessary keys to unlock a set of doors in a PDDL gridworld state.
Returns a vector of key sets for each door set, where each key set is a
vector of unique keys that can be used to unlock the doors.
"""
function find_necessary_keys(
    domain::Domain, state::State, doors::Vector{Const};
    unlockers = compute_unlockers(domain, state)
)
    key_sets = Vector{Const}[]
    for keys in Iterators.product((unlockers[door] for door in doors)...)
        key_set = unique(keys)
        length(key_set) == length(keys) || continue
        push!(key_sets, key_set)
    end
    return unique!(key_sets)
end

function find_necessary_keys(
    domain::Domain, state::State, door_sets::Vector{Vector{Const}};
    unlockers = compute_unlockers(domain, state)
)
    return [find_necessary_keys(domain, state, doors; unlockers)
            for doors in door_sets]
end

"""
    find_necessary_keys_and_doors(domain, state, r_start, r_goal, room_graph)
    find_necessary_keys_and_doors(domain, state, agent, item,
                                  room_labels, room_graph)

Finds the necessary keys and doors to traverse between two rooms in a PDDL
gridworld state. The first method takes room indices, and the second takes agent
and item PDDL objects.

Returns a vector of key/door sets, where each key/door set is a tuple of vectors
of keys and doors that must be traversed in order to reach the goal room. This 
vector is empty if there is no path between the rooms.
"""
function find_necessary_keys_and_doors(
    domain::Domain, state::State,
    r_start::Int, r_goal::Int, room_graph::AbstractGraph;
    all_doors = sort!(collect(PDDL.get_objects(state, :door)), by = x->x.name),
    unlockers = compute_unlockers(domain, state)
)
    door_sets = find_necessary_doors(domain, state, r_start, r_goal,
                                     room_graph; all_doors)
    iter = ((ks, ds) for ds in door_sets
            for ks in find_necessary_keys(domain, state, ds; unlockers))
    return collect(iter)
end

function find_necessary_keys_and_doors(
    domain::Domain, state::State,
    agent::Const, item::Const,
    room_labels::AbstractMatrix, room_graph::AbstractGraph;
    all_doors = sort!(collect(PDDL.get_objects(state, :door)), by=x->x.name),
    unlockers = compute_unlockers(domain, state)
)
    x, y = get_obj_loc(state, agent)
    r_start = room_labels[y, x]
    x, y = get_obj_loc(state, item, check_has=true)
    r_goal = room_labels[y, x]
    return find_necessary_keys_and_doors(domain, state, r_start, r_goal,
                                         room_graph; all_doors, unlockers)
end

"""
    find_necessary_keys_and_doors(domain, state, r_start, r_goals, room_graph)
    find_necessary_keys_and_doors(domain, state, agent, items,
                                  room_labels, room_graph)

Finds the necessary keys and doors to traverse between from a start location 
to a set of goal rooms in a PDDL gridworld state. The first method takes room
indices, and the second takes agent and item PDDL objects.

Returns a vector of key/door sets, where each key/door set is a tuple of vectors
of keys and doors that must be traversed in order to reach the goal rooms. This
vector is empty if there is no path between the rooms.
"""
function find_necessary_keys_and_doors(
    domain::Domain, state::State,
    r_start::Int, r_goals::AbstractVector{Int}, room_graph::AbstractGraph;
    all_doors = sort!(collect(PDDL.get_objects(state, :door)), by=x->x.name),
    unlockers = compute_unlockers(domain, state)
)
    @assert !isempty(r_goals) "No goal rooms specified."
    r_goals = unique!(r_goals)
    kd_sets = find_necessary_keys_and_doors(
        domain, state, r_start, r_goals[1], room_graph; all_doors, unlockers
    )
    length(r_goals) == 1 && return kd_sets
    for r in r_goals[2:end]
        new_kd_sets = find_necessary_keys_and_doors(
            domain, state, r_start, r, room_graph; all_doors, unlockers
        )
        tmp_kd_sets = Tuple{Vector{Const}, Vector{Const}}[]
        for (kd1, kd2) in Iterators.product(kd_sets, new_kd_sets)
            conflict = false
            for (k, d) in zip(kd2[1], kd2[2])
                k_idx = findfirst(==(k), kd1[1])
                d_idx = findfirst(==(d), kd1[2])
                k_idx == d_idx && continue # Same key-door pair
                isnothing(k_idx) && isnothing(d_idx) && continue # New pair
                conflict = true
                break
            end
            conflict && continue
            new_keys = union(kd1[1], kd2[1])
            new_doors = union(kd1[2], kd2[2])
            push!(tmp_kd_sets, (new_keys, new_doors))
        end
        kd_sets = tmp_kd_sets
    end
    return kd_sets
end

function find_necessary_keys_and_doors(
    domain::Domain, state::State,
    agent::Const, items::AbstractVector{Const},
    room_labels::AbstractMatrix, room_graph::AbstractGraph;
    all_doors = sort!(collect(PDDL.get_objects(state, :door)), by=x->x.name),
    all_agents = sort!(PDDL.get_objects(domain, state, :agent), by=x->x.name),
    unlockers = compute_unlockers(domain, state)
)
    x, y = get_obj_loc(state, agent)
    r_start = room_labels[y, x]
    r_goals = Int[]
    for item in items
        x, y = get_obj_loc(state, item)
        if (x < 0 || y < 0) # Handle held or unreachable items
            for a in all_agents
                if state[Compound(:has, Term[a, item])]
                    x, y = get_obj_loc(state, a)
                    break
                end
            end
            (x < 0 || y < 0) && continue # Item is unreachable
        end
        push!(r_goals, room_labels[y, x]::Int)
    end
    unique!(r_goals)
    if isempty(r_goals) || all(==(r_start), r_goals)
        return [(Const[], Const[])]
    end
    return find_necessary_keys_and_doors(domain, state, r_start, r_goals,
                                         room_graph; all_doors, unlockers)
end

"""
    find_necessary_keys_and_doors_recursive(
        domain, state, agent, items, room_labels, room_graph;
        max_iters = 5
    )

Recursively finds the necessary keys and doors to traverse between from a start
location to a set of goal rooms in a PDDL gridworld state. The process 
repeats until no new keys are added, or up to `max_iters`.
"""             
function find_necessary_keys_and_doors_recursive(
    domain::Domain, state::State,
    agent::Const, items::AbstractVector{Const},
    room_labels::AbstractMatrix, room_graph::AbstractGraph;
    max_iters::Int = 5,
    all_doors = sort!(collect(PDDL.get_objects(state, :door)), by=x->x.name),
    all_agents = sort!(PDDL.get_objects(domain, state, :agent), by=x->x.name),
    unlockers = compute_unlockers(domain, state)
)
    kd_sets = find_necessary_keys_and_doors(
        domain, state, agent, items, room_labels, room_graph;
        all_doors, all_agents, unlockers
    )
    length(kd_sets) == 1 && isempty(kd_sets[1][1]) && return kd_sets
    for _ in 1:max_iters
        new_kd_sets = nothing
        keys_added = false
        for (i, (old_keys, old_doors)) in enumerate(kd_sets)
            nec_keys = [k for (k, d) in zip(old_keys, old_doors)
                        if state[Compound(:locked, Term[d])]]
            isempty(nec_keys) && continue
            sub_kd_sets = find_necessary_keys_and_doors(
                domain, state, agent, nec_keys, room_labels, room_graph;
                all_doors, all_agents, unlockers
            )
            if !keys_added
                if length(sub_kd_sets) == 1 && isempty(sub_kd_sets[1][1])
                    continue
                else
                    new_kd_sets = kd_sets[1:i-1]
                    keys_added = true
                end
            end
            for (sub_keys, sub_doors) in sub_kd_sets
                conflict = false
                for (k, d) in zip(sub_keys, sub_doors)
                    k_idx = findfirst(==(k), old_keys)
                    d_idx = findfirst(==(d), old_doors)
                    k_idx == d_idx && continue # Same key-door pair
                    isnothing(k_idx) && isnothing(d_idx) && continue # New pair
                    conflict = true
                    break
                end
                conflict && continue
                new_keys = union(old_keys, sub_keys)
                new_doors = union(old_doors, sub_doors)
                push!(new_kd_sets, (new_keys, new_doors))
            end
            unique!(new_kd_sets)
        end
        !keys_added && break
        new_kd_sets == kd_sets && break
        kd_sets = new_kd_sets
    end
    return kd_sets
end

"""
    mst_estim_cost(domain, state, agents, goals, room_labels, room_graph,
                   loc_objects, loc_graph; kwargs...)
    mst_estim_cost(domain, state, agents, goals, key_door_sets,
                   loc_objects, loc_graph; kwargs...)
    mst_estim_cost(domain, state, agents, goals, plan_keys, plan_doors,
                   loc_objects, loc_graph; kwargs...)

Estimates the cost of achieving a set of goals in a PDDL gridworld state. The
cost is estimated by computing the minimum spanning tree of the graph induced
by the locations of the agents, goals, and plan objects.
"""
function mst_estim_cost(
    domain::Domain, state::State,
    agents::AbstractVector{Const},
    goals::AbstractVector{Const},
    room_labels::AbstractMatrix, room_graph::AbstractGraph,
    loc_objects::Vector{Const}, loc_graph::AbstractGraph;
    all_doors = sort!(collect(PDDL.get_objects(state, :door)), by=x->x.name),
    all_agents = sort!(PDDL.get_objects(domain, state, :agent), by=x->x.name),
    unlockers = compute_unlockers(domain, state),
    max_iters = 5,
    kwargs...
)
    agent = first(agents)
    to_reach = [goals; agents]
    kd_sets = find_necessary_keys_and_doors_recursive(
        domain, state, agent, to_reach, room_labels, room_graph;
        all_doors, unlockers, max_iters
    )
    return mst_estim_cost(domain, state, agents, goals, kd_sets,
                          loc_objects, loc_graph; kwargs...)
end

function mst_estim_cost(
    domain::Domain, state::State,
    agents::AbstractVector{Const},
    goals::AbstractVector{Const},
    kd_sets::AbstractVector{<:Tuple},
    loc_objects::Vector{Const}, loc_graph::AbstractGraph;
    kwargs...
)
    isempty(kd_sets) && return Inf
    return minimum(kd_sets) do (keys, doors)
        mst_estim_cost(domain, state, agents, goals, keys, doors,
                       loc_objects, loc_graph; kwargs...)
    end
end

function mst_estim_cost(
    domain::Domain, state::State,
    agents::AbstractVector{Const},
    goals::AbstractVector{Const},
    plan_keys::AbstractVector{Const},
    plan_doors::AbstractVector{Const},
    loc_objects::Vector{Const}, loc_graph::AbstractGraph;
    all_agents = sort!(PDDL.get_objects(domain, state, :agent), by=x->x.name)
)
    plan_keys = [k for (k, d) in zip(plan_keys, plan_doors)
                 if state[Compound(:locked, Term[d])]]
    plan_objects = [plan_keys; plan_doors; goals]
    all_helpers = filter(!in(agents), all_agents)
    cost = minimum(IterTools.subsets(all_helpers)) do helpers
        objects = [agents; helpers; plan_objects]
        plan_idxs = findall(in(objects), loc_objects)
        plan_subgraph, idx_map = induced_subgraph(loc_graph, plan_idxs)
        min_edges = kruskal_mst(plan_subgraph)
        if length(min_edges) < nv(plan_subgraph) - 1
            return Inf
        else
            total_weight = 0.0
            for edge in min_edges
                w = weight(edge)
                w <= eps() && continue
                total_weight += w
            end
            return total_weight
        end
    end
    return cost
end
