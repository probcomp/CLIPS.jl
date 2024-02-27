using DataStructures: OrderedDict
using Base: @kwdef

import SymbolicPlanners: compute, get_goal_terms

"""
    sys_sample_map!(f, items, probs, n_samples)

Applies `f` via systematic sampling of `items` with probabilities `probs`. For
each item, calls `f(item, frac)` where `frac` is the fraction of samples that
item should be included in. 
"""
function sys_sample_map!(f, items, probs, n_samples::Int)
    @assert length(items) == length(probs)
    @assert n_samples > 0
    count = 0
    total_prob = 0.0
    u = rand() / n_samples
    for (item, prob) in zip(items, probs)
        count == n_samples && break
        n_copies = 0
        total_prob += prob
        while u < total_prob
            n_copies += 1
            count += 1
            u += 1 / n_samples
        end
        n_copies == 0 && continue
        frac = n_copies / n_samples
        f(item, frac)
    end
    return nothing
end

"Returns the location of an item."
function get_item_loc(state::State, item::Const)
    for loc in PDDL.get_objects(state, :fixture)
        state[pddl"(placed $item $loc)"] && return loc
    end
    for loc in PDDL.get_objects(state, :agent)
        state[pddl"(holding $loc $item)"] && return loc
    end
    return pddl"(none)"
end

"Returns the location of an agent."
function get_agent_loc(state::State, agent::Const)
    for loc in PDDL.get_objects(state, :fixture)
        state[pddl"(next-to $agent $loc)"] && return loc
    end
    return pddl"(none)"
end

"Returns the type of an item."
function get_item_type(state::State, item::Const)
    objtype = PDDL.get_objtype(state, item)
    if objtype == :item
        for ty in PDDL.get_objects(state, :itype)
            state[pddl"(itemtype $item $ty)"] && return ty
        end
        return :item
    else
        return objtype
    end
end

"Rollout a planning solution to get a sequence of future actions."
function rollout_sol(
    domain::Domain, planner::Planner,
    state::State, sol::Solution, spec::Specification;
    max_steps::Int = 100
)
    # Special case handling of RTHS policies
    if planner isa RTHS
        planner = copy(planner.planner)
        planner.heuristic = PolicyValueHeuristic(sol)
        planner.h_mult = 2.0 # Rollout sub-optimal but satisificing plan
        search_sol = planner(domain, state, spec)
        if search_sol isa NullSolution
            return Vector{Compound}()
        else
            return collect(Compound, search_sol)
        end
    elseif sol isa NullSolution # If no solution, return empty vector
        return Vector{Compound}()
    else # Otherwise just rollout the policy greedily
        actions = Vector{Compound}()
        for _ in 1:max_steps
            act = best_action(sol, state)
            if ismissing(act) break end
            state = transition(domain, state, act)
            push!(actions, act)
            if is_goal(spec, domain, state) break end
        end
        return actions
    end
end

"Shortens a goal label."
function shorten_label(label::AbstractString)
    m = nothing
    if (m = match(r"^set_table(\d\w?)$", label)) !== nothing
        return "table$(m.captures[1])"
    elseif (m = match(r"^salad_(\w+)$", label)) !== nothing
        return first(m.captures[1]) * "salad"
    elseif (m = match(r"^stew_(\w+)$", label)) !== nothing
        return first(m.captures[1]) * "stew"
    else
        return label
    end
end

"Extract the items collected in an assistive plan."
function extract_assist_options(state::State, plan::AbstractVector{<:Term})
    all_options = filter(collect(PDDL.get_objects(state, :item))) do item
        loc = get_item_loc(state, item)
        loc == pddl"(table1)" && return false
        loc == pddl"(none)" && return false
        loc in PDDL.get_objects(state, :agent) && return false
        return true
    end
    options = Const[]
    for act in plan
        act.name != :grab && continue
        act.args[1].name != :helper && continue
        push!(options, act.args[2])
    end
    sort!(unique!(options), by=string)
    sort!(unique!(all_options), by=string)
    return options, all_options
end
