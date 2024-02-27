using SymbolicPlanners

import SymbolicPlanners:
    precompute!, is_precomputed, compute, filter_available, get_goal_terms

"""
    VirtualHomeHeuristic

Custom distance estimation and action filtering heuristic for
pick-and-place tasks in the VirtualHome domain.
"""
mutable struct VirtualHomeHeuristic <: Heuristic
    goal_items::Vector{Const}
    goal_fixtures::Vector{Const}
    goal_hash::Union{UInt, Nothing}
    VirtualHomeHeuristic() = new(Const[], Const[], nothing)
end

function precompute!(h::VirtualHomeHeuristic, domain::Domain, state::State)
    empty!(h.goal_items)
    append!(h.goal_items, PDDL.get_objects(state, :item))
    empty!(h.goal_fixtures)
    append!(h.goal_fixtures, PDDL.get_objects(state, :fixture))
    h.goal_hash = nothing
    return h
end

function precompute!(h::VirtualHomeHeuristic,
                     domain::Domain, state::State, spec::Specification)
    goal_terms = get_goal_terms(spec)
    goal = PDDL.dequantify(Compound(:and, goal_terms), domain, state)
    goal_terms = PDDL.flatten_conjs(goal)
    items = extract_items(goal_terms)
    append!(empty!(h.goal_items), items)
    fixtures = extract_fixtures(goal_terms)
    append!(empty!(h.goal_fixtures), fixtures)
    h.goal_hash = hash(goal_terms)
    return h
end

is_precomputed(h::VirtualHomeHeuristic) = isdefined(h, :goal_hash)

function compute(h::VirtualHomeHeuristic,
                 domain::Domain, state::State, spec::Specification)
    # If necessary, update action filter with new goal
    if isnothing(h.goal_hash) || h.goal_hash != hash(get_goal_terms(spec))
        precompute!(h, domain, state, spec)
    end
    # Compute number of agents who can grab items
    agents = PDDL.get_objects(state, :agent)
    n_nograb = sum(state[pddl"(nograb $a)"] for a in agents)
    n_cangrab = length(agents) - n_nograb
    # Estimate cost of picking up and placing items
    item_cost = sum(h.goal_items) do item
        loc = get_item_loc(state, item)
        if loc in h.goal_fixtures
            return 0.0f0
        elseif PDDL.get_objtype(state, loc) == :agent
            return 1.0f0
        elseif loc == pddl"(none)"
            return Inf32
        else
            return 1.0f0 + 1.0f0 * length(agents) / n_cangrab
        end
    end
    # Estimate cost of moving between locations
    relevant_fixtures = extract_relevant_fixtures(state, h.goal_items)
    relevant_fixtures = setdiff!(relevant_fixtures, h.goal_fixtures)
    if length(relevant_fixtures) > 0
        agent_locs = (get_agent_loc(state, a) for a in PDDL.get_objects(state, :agent))
        setdiff!(relevant_fixtures, agent_locs)
        move_cost = Float32(length(relevant_fixtures) + length(h.goal_fixtures))
    else
        move_cost = 0.0f0
        for agent in PDDL.get_objects(state, :agent)
            loc = get_agent_loc(state, agent)
            loc in h.goal_fixtures && continue
            if any(state[pddl"(holding $agent $i)"] for i in h.goal_items)
                move_cost += 1.0f0
            end
        end
    end
    return item_cost + move_cost
end

function filter_available(h::VirtualHomeHeuristic,
                          domain::Domain, state::State, spec::Specification)
    # If necessary, update action filter with new goal
    if isnothing(h.goal_hash) || h.goal_hash != hash(get_goal_terms(spec))
        precompute!(h, domain, state, spec)
    end
    # Compute relevant fixtures
    relevant_fixtures = extract_relevant_fixtures(state, h.goal_items)
    return Iterators.filter(available(domain, state)) do act
        agent = act.args[1]
        if state[Compound(:nograb, Term[agent])]
            if act.name == :move
                act.args[3] in h.goal_fixtures
            elseif act.name == :put && act.args[2] in h.goal_items
                act.args[3] in h.goal_fixtures
            else
                act.name == :wait
            end
        elseif act.name == :move
            act.args[3] in h.goal_fixtures || act.args[3] in relevant_fixtures
        elseif act.name == :grab
            act.args[2] in h.goal_items
        elseif act.name == :put && act.args[2] in h.goal_items
            act.args[3] in h.goal_fixtures
        elseif act.name == :stack
            false
        else
            true
        end
    end
end

"Extract goal items from a list of goal terms."
function extract_items(terms::AbstractVector{<:Term})
    items = Const[]
    for term in terms
        if term.name == :holding
            push!(items, term.args[2])
        elseif term.name == :placed
            push!(items, term.args[1])
        end
    end
    return unique!(items)
end
extract_items(term::Term) = extract_items(PDDL.flatten_conjs(term))

"Extract goal fixtures from a list of goal terms."
function extract_fixtures(terms::AbstractVector{<:Term})
    fixtures = Const[]
    for term in terms
        if term.name == :placed
            push!(fixtures, term.args[2])
        elseif term.name == Symbol("next-to")
            push!(fixtures, term.args[2])
        end
    end
    return unique!(fixtures)
end
extract_fixtures(term::Term) = extract_fixtures(PDDL.flatten_conjs(term))

"Extract relevant fixtures in the current state, given a list of goal items."
function extract_relevant_fixtures(state::State, items::AbstractVector{Const})
    fixtures = Const[]
    for item in items
        for f in PDDL.get_objects(state, :fixture)
            state[pddl"(placed $item $f)"] || continue
            push!(fixtures, f)
        end
    end
    return unique!(fixtures)
end
