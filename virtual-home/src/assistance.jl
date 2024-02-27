using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using Printf
using Random

using GenParticleFilters: softmax
using SymbolicPlanners: get_goal_terms, set_goal_terms, get_action_prob

## Dependencies ##

# include("utils.jl")
# include("heuristics.jl")

## Metrics ##

"Compute precision and recall for assistance options."
function compute_assist_metrics(
    assist_objs::Vector{Const},
    assist_option_probs::AbstractVector{<:Real},
    true_assist_options::Vector{Const};
    verbose::Bool = false
)
    idxs = Int[findfirst(==(o), assist_objs) for o in true_assist_options]
    assist_precision = sum(assist_option_probs[idxs]) / sum(assist_option_probs)
    assist_precision = isnan(assist_precision) ? 0.0 : assist_precision
    assist_recall = sum(assist_option_probs[idxs]) / length(true_assist_options)
    if verbose
        println("Option probabilities:")
        for (obj, prob) in zip(assist_objs, assist_option_probs)
            @printf("  %s: %.3f\n", obj, prob)
        end
        println()
        if !isempty(true_assist_options)
            @printf("Assist precision: %.3f\n", assist_precision)
            @printf("Assist recall: %.3f\n", assist_recall)
            println()
        end
    end
    return assist_precision, assist_recall
end

"Compute plan length, cost, and speaker cost."
function compute_plan_metrics(
    plan::AbstractVector{<:Term}, spec::Specification;
    speaker = pddl"(actor)", listener = pddl"(helper)", verbose::Bool = false
)
    verbose && println("\nComputing plan metrics...")
    plan_length = float(length(plan))
    plan_cost = sum(plan) do act
        act == PDDL.no_op && return 1.0
        return 1.0
    end
    speaker_cost = sum(plan) do act
        act == PDDL.no_op && return 1.0
        act.args[1] == listener && return 0.0
        return 1.0
    end
    if verbose
        @printf("Plan length: %.3f\n", plan_length)
        @printf("Plan cost: %.3f\n", plan_cost)
        @printf("Speaker cost: %.3f\n", speaker_cost)
    end
    return (
        plan_length = plan_length,
        plan_cost = plan_cost,
        speaker_cost = speaker_cost
    )
end

function compute_plan_metrics(
    plans::AbstractVector{<:AbstractVector{<:Term}}, spec::Specification,
    probs::AbstractVector{<:Real} = ones(length(plans)) ./ length(plans);
    verbose::Bool = false, kwargs...
)
    verbose && println("\nComputing average plan metrics...")
    mean_costs = sum(zip(plans, probs)) do (plan, prob)
        costs = compute_plan_metrics(plan, spec; verbose=false, kwargs...)
        return collect(Float64, costs) .* prob
    end
    names = (:plan_length, :plan_cost, :speaker_cost)
    mean_costs = NamedTuple{names}(Tuple(mean_costs))
    if verbose
        @printf("Plan length: %d\n", mean_costs.plan_length)
        @printf("Plan cost: %.3f\n", mean_costs.plan_cost)
        @printf("Speaker cost: %.3f\n", mean_costs.speaker_cost)
    end
    return mean_costs
end

## Literal Assistance ##

"""
    literal_assistance(command, domain, state, true_goal_spec; kwargs...)

Literal assistance model for a lifted `command` in a `domain` and
environment `state`. Computes the distribution over assistance options,
distribution over assistive plans, and the expected cost of those plans.
"""
function literal_assistance(
    command::ActionCommand,
    domain::Domain, state::State,
    true_goal_spec::Specification;
    true_assist_options = Const[],
    assist_objs = collect(PDDL.get_objects(state, :item)),
    speaker = pddl"(actor)",
    listener = pddl"(helper)",
    max_steps = 50,
    cmd_planner = AStarPlanner(VirtualHomeHeuristic(), max_nodes=2^16, verbose=true),
    goal_planner = AStarPlanner(VirtualHomeHeuristic(), max_nodes=2^16, verbose=true),
    verbose::Bool = false
)
    # Compute plan that satisfies command
    verbose && println("Planning for command: $command")
    cmd_goals = command_to_goals(command; speaker, listener)
    display(cmd_goals)
    cmd_goal_spec = set_goal_terms(true_goal_spec, cmd_goals)
    cmd_plan = nothing
    cmd_success = false
    for trial in (false, true) # Try planning with speaker frozen and unfrozen
        verbose && println("Speaker is frozen: $trial")
        tmp_state = copy(state)
        tmp_state[Compound(:frozen, Term[speaker])] = trial
        tmp_state[Compound(:nograb, Term[speaker])] = true
        cmd_sol = cmd_planner(domain, tmp_state, cmd_goal_spec)
        if cmd_sol isa NullSolution || cmd_sol.status != :success
            cmd_plan = Term[]
            verbose && println("No plan found.")
        else
            cmd_plan = collect(cmd_sol)
            cmd_success = length(cmd_plan) <= max_steps
            verbose && println("Plan found: $(length(cmd_plan)) actions")
            break
        end
    end
    
    # Compute remainder that satisfies speaker's true goal, freezing listener
    verbose && println("Planning for remainder...")
    cmd_end_state = isempty(cmd_plan) ?
        copy(state) : EndStateSimulator()(domain, state, cmd_plan)
    cmd_end_state[Compound(:nograb, Term[listener])] = true
    cmd_end_state[Compound(:nograb, Term[speaker])] = false
    if length(cmd_plan) < max_steps
        goal_sol = goal_planner(domain, cmd_end_state, true_goal_spec)
    else
        goal_sol = NullSolution(:max_depth)
    end
    if goal_sol isa NullSolution || goal_sol.status != :success
        goal_plan = fill(PDDL.no_op, max(max_steps - length(cmd_plan), 0))
        goal_success = false
        verbose && println("No plan found.")
    else
        goal_plan = collect(goal_sol)
        goal_success = (length(goal_plan) + length(cmd_plan)) <= max_steps
        verbose && println("Plan found: $(length(goal_plan)) actions")
    end
    full_plan = Term[cmd_plan; goal_plan]
    resize!(full_plan, min(max_steps, length(full_plan)))
    verbose && println()
    
    # Compute assistance options
    verbose && println("Computing assistance options...")
    assist_option_probs = zeros(length(assist_objs))
    focal_objs = extract_focal_objects(cmd_plan; listener)
    for obj in focal_objs
        obj_idx = findfirst(==(obj), assist_objs)
        isnothing(obj_idx) && continue
        assist_option_probs[obj_idx] += 1
    end

    # Calculate precision and recall from assistance option probabilities
    assist_precision, assist_recall =
        compute_assist_metrics(assist_objs, assist_option_probs,
                               true_assist_options; verbose)

    # Compute costs of assistance plans
    plan_metrics = compute_plan_metrics(full_plan, true_goal_spec;
                                        speaker, listener, verbose)
    if verbose
        @printf("Command success: %s\n", cmd_success)
        @printf("Goal success: %s\n", goal_success)
    end

    return (
        assist_objs = assist_objs,
        assist_option_probs = assist_option_probs,
        assist_precision = assist_precision,
        assist_recall = assist_recall,
        plan_metrics...,
        cmd_success = float(cmd_success),
        goal_success = float(goal_success),
        cmd_plan = cmd_plan,
        full_plan = full_plan
    )
end

"""
    literal_assistance(commands, probs, domain, state, true_goal_spec;
                       n_samples=10, kwargs...)

Literal assistance model for a distribution of lifted `commands` in a
`domain` and environment `state`. Uses systematic sampling with `n_samples`
to compute the expected distribution over assistance options and the expected
cost of the assistive plans.
"""
function literal_assistance(
    commands::AbstractVector{ActionCommand},
    probs::AbstractVector{<:Real},
    domain::Domain, state::State,
    true_goal_spec::Specification;
    true_assist_options = Const[],
    assist_objs = collect(PDDL.get_objects(state, :item)),
    n_samples::Int = 10,
    verbose = false,
    kwargs...
)
    # Set up containers
    assist_option_probs = zeros(length(assist_objs))
    sample_probs = Float64[]
    result_samples = (
        plan_length = Float64[],
        plan_cost = Float64[], speaker_cost = Float64[],
        cmd_success = Float64[], goal_success = Float64[]
    )

    # Compute expected assistance options and costs via systematic sampling
    verbose && println("Computing expected values via systematic sampling...")
    sys_sample_map!(commands, probs, n_samples) do command, prob
        verbose && println("Sampling command: $command")
        result = literal_assistance(
            command, domain, state, true_goal_spec;
            verbose, true_assist_options, assist_objs, kwargs...
        )
        assist_option_probs .+= result.assist_option_probs .* prob
        for (field, vals) in pairs(result_samples)
            push!(vals, result[field])
        end
        push!(sample_probs, prob)
        verbose && println()
    end
    mean_results = map(vs -> vs' * sample_probs, result_samples)

    if verbose
        println()
        println("== Expected values ==")
    end

    # Calculate precision and recall from assistance option probabilities
    assist_precision, assist_recall =
        compute_assist_metrics(assist_objs, assist_option_probs,
                               true_assist_options; verbose)

    if verbose
        @printf("Plan length: %.2f\n", mean_results.plan_length)
        @printf("Plan cost: %.2f\n", mean_results.plan_cost)
        @printf("Speaker cost: %.2f\n", mean_results.speaker_cost)
        @printf("Command success rate: %.2f\n", mean_results.cmd_success)
        @printf("Goal success rate: %.2f\n", mean_results.goal_success)
    end

    return (
        assist_objs = assist_objs,
        assist_option_probs = assist_option_probs,
        assist_precision = assist_precision,
        assist_recall = assist_recall,
        mean_results...,
        sample_probs = sample_probs,
    )
end

## Pragmatic Assistance ##

"""
    pragmatic_assistance_pmdp(
        pf, domain, state, true_goal_spec;
        true_plan = Term[], true_assist_options = Const[], 
        estim_type = "mean", kwargs...
    )

Pragmatic assistance by following one or more of the inferred joint policies
associated with the inferred goal specifications in an open-loop manner. The
assistive policy ̄π-MDP is thus a mixture over the policies π associated
with each underlying multi-agent MDP. If `estim_type` is set to `"mode"`, then
the policy π*-MDP for the most likely goal is used instead.

To calculate the cost of an assistive plan, the assistant's actions are
executed according to the joint policy for one of the goal specifications,
while the principal's actions are simulated according to the `true_plan`.
Once the principal detects that the assistant is not cooperating, the principal
switches to acting alone. The cost of the resulting plan is then averaged
across all goal specifications.

Returns the distribution over assistance options, the assistive plan for 
the most likely goal, and the cost of that plan.
"""
function pragmatic_assistance_pmdp(
    pf::ParticleFilterState,
    domain::Domain, state::State,
    true_goal_spec::Specification;
    true_plan::AbstractVector{<:Term} = Term[],
    true_assist_options = Const[],
    assist_objs = collect(PDDL.get_objects(state, :item)),
    estim_type::AbstractString = "mean",
    speaker = pddl"(actor)",
    listener = pddl"(helper)",
    max_steps::Int = 50,
    n_iters = 0,
    max_nodes = 2^16,
    max_time = 10.0,
    p_thresh::Float64 = 0.02,
    coop_log_odds_thresh::Float64 = -log(10),
    coop_act_temperature::Float64 = 1.0,
    verbose::Bool = false
)
    # Extract probabilities, specifications and policies from particle filter
    start_t = InversePlanning.get_model_timestep(pf)
    probs = get_norm_weights(pf)
    goal_specs = map(pf.traces) do trace
        trace[:init => :agent => :goal]
    end
    policies = map(pf.traces) do trace
        if start_t == 0
            copy(trace[:init => :agent => :plan].sol)
        else
            copy(trace[:timestep => start_t => :agent => :plan].sol)
        end
    end

    # Construct planner to refine policies in unvisited states
    heuristic = precomputed(VirtualHomeHeuristic(), domain, state)
    planner = RTHS(heuristic=heuristic, n_iters=1,
                   max_nodes=max_nodes, max_time=max_time, fail_fast=true)

    # Initialize speaker's policy under true goal
    planner.n_iters = 0
    true_idx = findfirst(==(true_goal_spec), goal_specs)
    true_policy = isnothing(true_idx) ?
        planner(domain, state, true_goal_spec) : policies[true_idx]
    true_trajectory = PDDL.simulate(domain, state, true_plan)
    planner.n_iters = n_iters

    # Retain most probable goals and policies
    if estim_type == "mode"
        max_prob = maximum(probs)
        max_idxs = findall(==(max_prob), probs)
        probs = ones(length(max_idxs)) ./ length(max_idxs)
        goal_specs = goal_specs[max_idxs]
        policies = policies[max_idxs]
    end

    # Compute assistance options by rolling out policies for each goal
    verbose && println("Computing assistance options...")
    assist_option_probs = zeros(length(assist_objs))

    for (prob, spec, policy) in zip(probs, goal_specs, policies)
        plan = rollout_sol(domain, planner, state, policy, spec;
                           max_steps = max_steps-start_t)
        listener_plan = filter(act -> act.args[1] == listener, plan)
        focal_objs = extract_focal_objects(listener_plan)
        for obj in focal_objs
            obj_idx = findfirst(==(obj), assist_objs)
            isnothing(obj_idx) && continue
            assist_option_probs[obj_idx] += prob
        end
    end

    # Calculate precision and recall from assistance option probabilities
    assist_precision, assist_recall =
        compute_assist_metrics(assist_objs, assist_option_probs,
                               true_assist_options; verbose)

    # Construct policies to monitor if assistant is cooperating
    soft_policy = BoltzmannPolicy(true_policy, coop_act_temperature)
    rand_policy = RandomPolicy(domain)

    # Iteratively take actions according to each policy
    assist_plans = Vector{Term}[]
    goal_success = 0.0
    init_state = copy(state)
    for (prob, spec, policy) in zip(probs, goal_specs, policies)
        if prob < p_thresh # Ignore low-probability goals
            assist_plan = Term[PDDL.no_op for _ in (start_t+1):max_steps]
            push!(assist_plans, assist_plan)
            continue
        end
        coop_log_odds = 0.0 # Monitor whether assistant is cooperating
        go_solo = false # Flag for whether principal is acting solo
        state = copy(init_state)
        goal_str = join(write_pddl.(get_goal_terms(spec)), " ")
        verbose && println("Planning for $goal_str (p = $prob)...")
        assist_plan = Term[]
        for t in (start_t+1):max_steps
            # If assistant is not cooperating, freeze assistant and act solo
            if !go_solo && coop_log_odds < coop_log_odds_thresh
                state[Compound(:frozen, Term[listener])] = true
                go_solo = true
            end
            # Refine speaker's policy to true goal
            true_policy = 
                refine!(true_policy, planner, domain, state, true_goal_spec)
            cur_agent =
                state[Compound(:active, Term[speaker])] ? speaker : listener
            # Speaker acts
            if state[Compound(:active, Term[speaker])] || go_solo
                t_true = findfirst(==(state), true_trajectory)
                if !go_solo && !isnothing(t_true) && t_true <= length(true_plan)
                    # Try to take action from true plan
                    act = true_plan[t_true]
                elseif SymbolicPlanners.get_value(true_policy, state) == -Inf
                    # If true goal is unreachable, wait
                    act = Compound(:wait, Term[cur_agent])
                else # Select best action according to policy
                    act = SymbolicPlanners.best_action(true_policy, state)
                    if ismissing(act)
                        act = Compound(:wait, Term[cur_agent])
                    end
                end
            else # Listener acts
                policy = refine!(policy, planner, domain, state, spec)
                # Take best action if goal is still reachable
                if SymbolicPlanners.get_value(policy, state) > -Inf
                    act = SymbolicPlanners.best_action(policy, state)
                    if ismissing(act)
                        act = Compound(:wait, Term[listener])
                    end
                else
                    act = Compound(:wait, Term[listener])
                end
                # Update cooperation log odds
                if !go_solo
                    p_act_true = get_action_prob(soft_policy, state, act)
                    p_act_rand = get_action_prob(rand_policy, state, act)
                    coop_log_odds += log(p_act_true) - log(p_act_rand)
                end
            end
            push!(assist_plan, act)
            state = transition(domain, state, act; check=true)
            if verbose 
                print("$(t-start_t)\t$(rpad(write_pddl(act), 30))")
                @printf("coop log odds = %.2f\n", coop_log_odds)
            end
            # Check if goal is satisfied
            if is_goal(true_goal_spec, domain, state)
                goal_success += prob
                verbose && println("Goal satisfied at timestep $(t-start_t).")
                break
            end
        end
        push!(assist_plans, assist_plan)
    end

    # Compute plan metrics
    plan_metrics = compute_plan_metrics(assist_plans, true_goal_spec, probs;
                                        speaker, listener, verbose)
    verbose && @printf("Goal success: %s\n", goal_success)

    if estim_type == "mode"
        assist_plan = first(assist_plans)
    else
        max_idx = argmax(probs)
        assist_plan = assist_plans[max_idx]
    end

    return (
        assist_objs = assist_objs,
        assist_option_probs = assist_option_probs,
        assist_precision = assist_precision,
        assist_recall = assist_recall,
        plan_metrics...,
        goal_success = goal_success,
        full_plan = assist_plan
    )
end

"""
    pragmatic_assistance_qmdp_act(
        pf, domain, state, true_goal_spec;
        true_plan = Term[], true_assist_options = Const[], kwargs...
    )

Pragmatic assistance using the Q-MDP approximation of the corresponding
assistive POMDP. Given a particle filter belief state (`pf`) in a `domain` and
environment `state`, simulates the best assistive action to take at each step 
by maximizing the expected Q-value, where the expectation is taken over
goal specifications, and maximization is over actions.

After each assistant's action, the principal takes an action
according to either the `true_plan` (if possible), or by following the 
optimal policy to the true goal. The assistant then takes another action 
at the resulting state.  Once the principal detects that the assistant is not
cooperating, the principal switches to acting alone.

Returns the distribution over assistance options, the assistive plan,
and the cost of that plan.
"""
function pragmatic_assistance_qmdp_act(
    pf::ParticleFilterState,
    domain::Domain, state::State,
    true_goal_spec::Specification;
    true_plan::AbstractVector{<:Term} = Term[],
    true_assist_options = Const[],
    assist_objs = collect(PDDL.get_objects(state, :item)),
    update_beliefs::Bool = false,
    sips_config::Union{SIPS, Nothing} = nothing,
    speaker = pddl"(actor)",
    listener = pddl"(helper)",
    max_steps::Int = 50,
    p_thresh::Float64 = 0.02,
    n_iters = 0,
    max_nodes = 2^16,
    max_time = 10.0,
    coop_log_odds_thresh::Float64 = -log(10),
    coop_act_temperature::Float64 = 1.0,
    verbose::Bool = false
)
    # Extract probabilities, specifications and policies from particle filter
    start_t = InversePlanning.get_model_timestep(pf)
    goal_specs = map(pf.traces) do trace
        trace[:init => :agent => :goal]
    end
    if !update_beliefs
        probs = get_norm_weights(pf)
        policies = map(pf.traces) do trace
            if start_t == 0
                copy(trace[:init => :agent => :plan].sol)
            else
                copy(trace[:timestep => start_t => :agent => :plan].sol)
            end
        end
    else
        pf = copy(pf)
        goal_addr = :init => :agent => :goal => :goal
        goal_probs = proportionmap(pf, goal_addr)
        print_cb = PrintStatsCallback((goal_addr, 1:length(goal_probs)))   
    end

    # Initialize speaker's policy under true goal
    heuristic = precomputed(VirtualHomeHeuristic(), domain, state)
    planner = RTHS(heuristic=heuristic, n_iters=0,
                   max_nodes=max_nodes, max_time=max_time, fail_fast=true)
    true_idx = update_beliefs ? 
        nothing : findfirst(==(true_goal_spec), goal_specs)
    true_policy = isnothing(true_idx) ?
        planner(domain, state, true_goal_spec) : policies[true_idx]
    true_trajectory = PDDL.simulate(domain, state, true_plan)
    planner.n_iters = n_iters

    # Construct policies to monitor if assistant is cooperating
    soft_policy = BoltzmannPolicy(true_policy, coop_act_temperature)
    rand_policy = RandomPolicy(domain)
    coop_log_odds = 0.0 
    go_solo = false # Flag for whether principal is acting solo

    # Iteratively take action that minimizes expected goal achievement cost
    verbose && println("Planning future actions via pragmatic assistance...")
    assist_plan = Term[]
    goal_success = false
    state = copy(state)
    for t in (start_t+1):max_steps
        cur_agent =
            state[Compound(:active, Term[speaker])] ? speaker : listener
        # If assistant is not cooperating, freeze assistant and act solo
        if coop_log_odds < coop_log_odds_thresh
            go_solo = update_beliefs ? false : true
            speaker_state = copy(state)
            speaker_state[Compound(:frozen, Term[listener])] = true
            # Refine policies for both true state and speaker-only state
            if update_beliefs
                refine!(true_policy, planner, domain, state, true_goal_spec)
            end
            refine!(true_policy, planner, domain, speaker_state, true_goal_spec)
        else
            # Refine policy for only true state
            refine!(true_policy, planner, domain, state, true_goal_spec)
            speaker_state = state
        end
        # Speaker acts
        if state[Compound(:active, Term[speaker])] || go_solo
            t_true = findfirst(==(speaker_state), true_trajectory)
            t_cur = t - start_t
            if !go_solo && ((!isnothing(t_true) && t_true <= length(true_plan)) ||
                            (t_cur <= length(true_plan) && available(domain, speaker_state, true_plan[t_cur])))
                # Try to take action from true plan
                if !isnothing(t_true)
                    act = true_plan[t_true]
                else
                    act = true_plan[t_cur]
                end
            elseif SymbolicPlanners.get_value(true_policy, speaker_state) == -Inf
                # If true goal is unreachable, wait
                act = Compound(:wait, Term[cur_agent])
            else # Select best action according to policy
                act = SymbolicPlanners.best_action(true_policy, speaker_state)
                if ismissing(act)
                    act = Compound(:wait, Term[cur_agent])
                end
            end
        else # Listener acts
            # Refine inferred policies for each goal
            if update_beliefs
                probs = get_norm_weights(pf)
                policies = map(pf.traces) do trace
                    if t-1 == 0
                        copy(trace[:init => :agent => :plan].sol)
                    else
                        copy(trace[:timestep => t-1 => :agent => :plan].sol)
                    end
                end
            end
            best_acts = Term[]
            for (prob, policy, spec) in zip(probs, policies, goal_specs)
                prob < p_thresh && continue # Ignore low-probability goals
                policy = refine!(policy, planner, domain, state, spec)
                push!(best_acts, SymbolicPlanners.best_action(policy, state))
            end
            # Compute Q-values for each action
            act_values = Dict{Term, Float64}()
            for act in available(domain, state)
                next_state = transition(domain, state, act; check=true)
                act_values[act] = 0.0
                agent = act.args[1]
                wait = Compound(:wait, Term[agent])
                # Compute expected value / probability of each action across goals
                for (prob, policy, spec) in zip(probs, policies, goal_specs)
                    prob < p_thresh && continue # Ignore low-probability goals
                    # Compute expected value of listener actions
                    val = SymbolicPlanners.get_value(policy, state, act)
                    # Ensure Q-values are finite
                    wait_cost = get_cost(spec, domain, state, wait, state)
                    act_cost = get_cost(spec, domain, state, act, next_state)
                    min_val = -((max_steps - t) * wait_cost + act_cost)
                    val = max(val, min_val)
                    act_values[act] += prob * val
                end
            end
            # Take action with highest value
            act = all(a == best_acts[1] for a in best_acts) ?
                best_acts[1] : argmax(act_values)
            # Update cooperation log odds
            p_act_true = get_action_prob(soft_policy, state, act)
            p_act_rand = get_action_prob(rand_policy, state, act)
            coop_log_odds += log(p_act_true) - log(p_act_rand)
        end
        # Transition to next state
        push!(assist_plan, act)
        state = transition(domain, state, act; check=true)
        if verbose 
            print("$(t-start_t)\t$(rpad(write_pddl(act), 30))")
            @printf("coop log odds = %.2f\n", coop_log_odds)
        end
        # Update particle filter belief state 
        if update_beliefs
            obs_act = cur_agent == listener ? InversePlanning.do_op(act) : act
            obs = choicemap(
                (:timestep => t => :act => :act, obs_act),
                (:timestep => t => :act => :speak, false),
            )
            pf = sips_step!(pf, sips_config, t, obs; callback=print_cb)
        end
        # Check if goal is satisfied
        if is_goal(true_goal_spec, domain, state)
            goal_success = true
            verbose && println("Goal satisfied at timestep $(t-start_t).")
            break
        end
    end
    
    # Extract assistance options
    verbose && println("\nComputing assistance options...")
    assist_option_probs = zeros(length(assist_objs))
    listener_plan = filter(act -> act.args[1] == listener, assist_plan)
    focal_objs = extract_focal_objects(listener_plan)
    for obj in focal_objs
        obj_idx = findfirst(==(obj), assist_objs)
        isnothing(obj_idx) && continue
        assist_option_probs[obj_idx] += 1
    end

    # Calculate precision and recall from assistance option probabilities
    assist_precision, assist_recall =
        compute_assist_metrics(assist_objs, assist_option_probs,
                               true_assist_options; verbose)

    # Compute plan metrics
    plan_metrics = compute_plan_metrics(assist_plan, true_goal_spec;
                                        speaker, listener, verbose)
    verbose && @printf("Goal success: %s\n", goal_success)

    return (
        assist_objs = assist_objs,
        assist_option_probs = assist_option_probs,
        assist_precision = assist_precision,
        assist_recall = assist_recall,
        plan_metrics...,
        goal_success = goal_success,
        full_plan = assist_plan,
    )
end

"""
    pragmatic_assistance_qmdp_plan(
        pf, domain, state, true_goal_spec;
        true_assist_options = Const[], estim_type = "mode",
        rerank_temperature = 1.0, kwargs...
    )

Pragmatic assistance using the Q-MDP approximation at the plan level.
Given a particle filter belief state (`pf`) in a `domain` and
environment `state`, computes the optimal plan for each goal, then estimates
the expected value of each plan across goals.

If `estim_type` is `"mode"`, then the plan with the lowest expected cost is 
chosen. Otherwise, if `estim_type` is `"mean"`, then the plans are weighted 
according to a Boltzmann distribution over their expected costs, and 
plan metrics are computed accordingly.
"""
function pragmatic_assistance_qmdp_plan(
    pf::ParticleFilterState,
    domain::Domain, state::State,
    true_goal_spec::Specification;
    true_assist_options = Const[],
    assist_objs = collect(PDDL.get_objects(state, :item)),
    speaker = pddl"(actor)",
    listener = pddl"(helper)",
    max_steps::Int = 50,
    estim_type = "mode",
    n_iters = 0,
    max_nodes = 2^16,
    max_time = 60.0,
    rerank_temperature::Float64 = 1.0,
    verbose::Bool = false
)
    # Extract probabilities, specifications and policies from particle filter
    start_t = InversePlanning.get_model_timestep(pf)
    probs = get_norm_weights(pf)
    goal_specs = map(pf.traces) do trace
        trace[:init => :agent => :goal]
    end
    policies = map(pf.traces) do trace
        if start_t == 0
            copy(trace[:init => :agent => :plan].sol)
        else
            copy(trace[:timestep => start_t => :agent => :plan].sol)
        end
    end

    # Initialize speaker's policy under true goal
    heuristic = precomputed(VirtualHomeHeuristic(), domain, state)
    planner = RTHS(heuristic=heuristic, n_iters=0,
                   max_nodes=max_nodes, max_time=max_time, fail_fast=true)
    true_idx = findfirst(==(true_goal_spec), goal_specs)
    true_policy = isnothing(true_idx) ?
        planner(domain, state, true_goal_spec) : policies[true_idx]
    planner.n_iters = n_iters

    # Rollout each inferred policy to produce a plan
    assist_plans = map(zip(goal_specs, policies)) do (spec, policy)
        plan = rollout_sol(domain, planner, state, policy, spec;
                           max_steps = max_steps-start_t)
        last_listener_idx = findlast(plan) do act
            act.args[1] == listener && act.name != :wait
        end
        last_listener_idx === nothing && return empty(plan)
        return plan[1:last_listener_idx]
    end

    # Estimate expected (negated) return of each plan
    est_plan_costs = map(assist_plans) do plan
        remain_steps = max(max_steps - start_t - length(plan), 0)
        plan_end_state = PDDL.simulate(EndStateSimulator(), domain, state, plan)
        plan_end_state[Compound(:frozen, Term[listener])] = true
        cost = 0.0
        for (prob, spec, policy) in zip(probs, goal_specs, policies)
            cost -= prob .* PDDL.simulate(RewardAccumulator(),
                                          domain, state, plan, spec)
            plan_end_val = -heuristic(domain, plan_end_state, spec)
            if plan_end_val > -Inf
                refine!(policy, planner, domain, plan_end_state, spec)
                plan_end_val = SymbolicPlanners.get_value(policy, plan_end_state)
            end
            if plan_end_val == -Inf
                plan_end_val = -1.0 * remain_steps
            end
            cost -= prob * plan_end_val
        end
        return cost
    end

    if verbose
        println("Expected plan costs:")
        for (idx, (spec, cost)) in enumerate(zip(goal_specs, est_plan_costs))
            goal_str = join(write_pddl.(get_goal_terms(spec)), " ")
            @printf("%d. %s:\t%.2f\n", idx, goal_str, cost)
        end
        println()
    end

    # Select best plans
    if estim_type == "mode"
        best_plan_cost = minimum(est_plan_costs)
        best_idxs = findall(==(best_plan_cost), est_plan_costs)
        assist_plans = assist_plans[best_idxs]
        est_plan_costs = est_plan_costs[best_idxs]
    end

    # Compute plan probabilities via a Boltzmann distribution over their values
    assist_plan_probs = softmax(-est_plan_costs ./ rerank_temperature)

    # Compute assistance options and average across plans
    verbose && println("Computing assistance options...")
    assist_option_probs = zeros(length(assist_objs))

    for (prob, plan) in zip(assist_plan_probs, assist_plans)
        listener_plan = filter(act -> act.args[1] == listener, plan)
        focal_objs = extract_focal_objects(listener_plan)
        for obj in focal_objs
            obj_idx = findfirst(==(obj), assist_objs)
            isnothing(obj_idx) && continue
            assist_option_probs[obj_idx] += prob
        end
    end

    # Calculate precision and recall from assistance option probabilities
    assist_precision, assist_recall =
        compute_assist_metrics(assist_objs, assist_option_probs,
                               true_assist_options; verbose)

    # Complete each plan assuming that principal acts alone
    goal_successes = Float64[]
    assist_plans = map(assist_plans) do plan
        remain_steps = max(max_steps - start_t - length(plan), 0)
        plan_end_state = PDDL.simulate(EndStateSimulator(), domain, state, plan)
        plan_end_state[Compound(:frozen, Term[listener])] = true
        refine!(true_policy, planner, domain, plan_end_state, true_goal_spec)
        if SymbolicPlanners.get_value(true_policy, plan_end_state) == -Inf
            completion = fill(PDDL.no_op, remain_steps)
        else
            completion = rollout_sol(domain, planner, plan_end_state,
                                     true_policy, true_goal_spec;
                                     max_steps = remain_steps)
        end
        end_state = PDDL.simulate(EndStateSimulator(), domain,
                                  plan_end_state, completion)
        goal_success = is_goal(true_goal_spec, domain, end_state)
        push!(goal_successes, goal_success ? 1.0 : 0.0)
        return vcat(plan, completion)
    end

    # Compute plan metrics
    goal_success = mean(goal_successes)
    plan_metrics = compute_plan_metrics(
        assist_plans, true_goal_spec, assist_plan_probs;
        speaker, listener, verbose
    )
    verbose && @printf("Goal success: %s\n", goal_success)

    if estim_type == "mode"
        assist_plan = first(assist_plans)
    else
        max_idx = argmax(assist_plan_probs)
        assist_plan = assist_plans[max_idx]
    end

    return (
        assist_objs = assist_objs,
        assist_option_probs = assist_option_probs,
        assist_precision = assist_precision,
        assist_recall = assist_recall,
        plan_metrics...,
        goal_success = goal_success,
        full_plan = assist_plan,
    )
end
