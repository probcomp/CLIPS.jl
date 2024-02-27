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

"Returns the color of an object."
function get_obj_color(state::State, obj::Const)
    for color in PDDL.get_objects(state, :color)
        if state[Compound(:iscolor, Term[obj, color])]
            return color
        end
    end
    return Const(:none)
end

"Returns the location of an object."
function get_obj_loc(state::State, obj::Const; check_has::Bool=false)
    x = state[Compound(:xloc, Term[obj])]::Int
    y = state[Compound(:yloc, Term[obj])]::Int
    # Check if object is held by an agent, and return agent's location if so
    if check_has && PDDL.get_objtype(state, obj) in (:gem, :key)
        for agent in PDDL.get_objects(state, :human)
            if state[Compound(:has, Term[agent, obj])]
                x, y = get_obj_loc(state, agent)
                break
            end
        end
        for agent in PDDL.get_objects(state, :robot)
            if state[Compound(:has, Term[agent, obj])]
                x, y = get_obj_loc(state, agent)
                break
            end
        end
    end
    return (x, y)
end

"Rollout a planning solution to get a sequence of future actions."
function rollout_sol(
    domain::Domain, planner::Planner,
    state::State, sol::Solution, spec::Specification;
    max_steps::Int = 100
)
    # Special case handling of RTHS policies without reusable trees
    if planner isa RTHS && sol isa TabularVPolicy
        heuristic = planner.heuristic
        planner.heuristic = PolicyValueHeuristic(sol)
        search_sol = planner.planner(domain, state, spec)
        planner.heuristic = heuristic
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

"Extract the goal object and goal term achieved by a plan."
function extract_goal(plan::AbstractVector{<:Term})
    @assert plan[end].name == :pickup
    goal_agent = plan[end].args[1]::Const
    goal_obj = plan[end].args[2]::Const
    return goal_obj, Compound(:has, Term[goal_agent, goal_obj])
end

"Extract the keys or doors collected in an assistive plan."
function extract_assist_options(state::State, plan::AbstractVector{<:Term},
                                assist_type::AbstractString)
    options = Const[]
    if assist_type == "keys"
        option_count = length(PDDL.get_objects(state, :key))
    elseif assist_type == "doors"
        option_count = length(PDDL.get_objects(state, :door))
    else
        error("Unknown assist type: $assist_type")
    end
    for act in plan
        if assist_type == "keys"
            act.name != :pickup && continue
            act.args[1].name == :human && continue
            obj = act.args[2]
            PDDL.get_objtype(state, obj) == :key || continue
        elseif assist_type == "doors"
            act.name != :unlock && continue
            act.args[1].name == :human && continue
            obj = act.args[3]
            PDDL.get_objtype(state, obj) == :door || continue
        else
            error("Unknown assist type: $assist_type")
        end
        push!(options, obj)
    end
    return sort!(unique!(options), by=string), option_count
end

"Decompose goals into DNF clauses."
function decompose_goals(domain::Domain, state::State, spec::Specification)
    goals = get_goal_terms(spec)
    return decompose_goals(domain, state, goals)
end

function decompose_goals(domain::Domain, state::State,
                         goals::AbstractVector{<:Term})
    if goals[1].name == Symbol("do-action") # Handle action goals
        return transform_action_goal(domain, state, goals[1])
    else # Handle regular goals
        new_goals = transform_goals(goals)
        return PDDL.to_dnf_clauses(Compound(:and, unique!(new_goals)))
    end
end

"Transform action goals into DNF clauses."
function transform_goals(goals::Vector{Term})
    new_goals = Term[]
    for goal in goals
        if goal.name in (:has, Symbol("unlocked-by"))
            push!(new_goals, goal)
        elseif goal.name in (:and, :or)
            subgoals = transform_goals(goal.args)
            push!(new_goals, Compound(goal.name, subgoals))
        end
    end
    return new_goals
end

"Transform action goals into DNF clauses."
function transform_action_goal(domain::Domain, state::State, action_goal::Term)
    action = action_goal.args[1]
    new_goals = Term[]
    if PDDL.is_ground(action)
        ground_actions = [action]
    else
        constraints = action_goal.args[2]
        substs = satisfiers(domain, state, constraints)
        ground_actions = [PDDL.substitute(action, s) for s in substs]
    end
    for act in ground_actions
        if act.name == :pickup
            # Pickup cost is equivalent to cost of agent having the item
            agent = act.args[1]
            item = act.args[2]
            push!(new_goals, Compound(:has, Term[agent, item]))
        elseif action.name == :handover
            # Handover cost is underestimated by assuming either agent has item
            a1, a2 = act.args[1:2]
            item = act.args[3]
            push!(new_goals, Compound(:has, Term[a1, item]))
            push!(new_goals, Compound(:has, Term[a2, item]))
        elseif action.name == :unlock
            # Unlock cost is underestimated by the cost of agent having the key
            agent = act.args[1]
            key = act.args[2]
            push!(new_goals, Compound(:has, Term[agent, key]))
        end
    end
    return new_goals    
end

"Extract action costs from a specification."
function extract_action_costs(spec::Specification, agents::Vector{Const})
    move_costs = Float64[]
    wait_costs = Float64[]
    pickup_costs = Float64[]
    unlock_costs = Float64[]
    for agent in agents
        if !has_action_cost(spec)
            push!(move_costs, 1.0)
            push!(wait_costs, 1.0)
            push!(pickup_costs, 1.0)
            push!(unlock_costs, 1.0)
            continue
        end
        move_cost = minimum((:up, :down, :left, :right)) do act
            get_action_cost(spec, Compound(act, Term[agent]))
        end
        wait_cost =
            get_action_cost(spec, Compound(:wait, Term[agent]))
        pickup_cost =
            get_action_cost(spec, Compound(:pickup, [agent, Var(:X)]))
        unlock_cost =
            get_action_cost(spec, Compound(:unlock, [agent, Var(:X), Var(:Y)]))
        push!(move_costs, move_cost)
        push!(wait_costs, wait_cost)
        push!(pickup_costs, pickup_cost)
        push!(unlock_costs, unlock_cost)
    end
    return (move_costs, wait_costs, pickup_costs, unlock_costs)
end

"Extract action costs from a specification."
function extract_min_action_costs(spec::Specification, agents::Vector{Const})
    move_costs, wait_costs, pickup_costs, unlock_costs =
        extract_action_costs(spec, agents)
    return (minimum(move_costs), minimum(wait_costs),
            minimum(pickup_costs), minimum(unlock_costs))
end
