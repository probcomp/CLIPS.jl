using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using Printf
using Random

using GenParticleFilters: softmax
using SymbolicPlanners: get_goal_terms, set_goal_terms

## Dependencies ##

# include("utils.jl")
# include("heuristics.jl")
# include("utterances.jl")

## Literal Inference ##

"""
    literal_command_inference(domain, state, utterance)

Runs literal listener inference for an `utterance` in a `domain` and environment 
`state`. Returns a distribution over listener-directed commands that could
have led to the utterance.
"""
function literal_command_inference(
    domain::Domain, state::State, utterance::String;
    speaker = pddl"(human)",
    listener = pddl"(robot)",
    verbose::Bool = false
)
    # Enumerate over listener-directed commands
    verbose && println("Enumerating commands...")
    actions, agents, predicates =
        enumerate_salient_actions(domain, state; salient_agents=[listener])
    commands =
        enumerate_commands(actions, agents, predicates; speaker, listener)
    # Lift commands and remove duplicates
    commands = lift_command.(commands, [state])
    unique!(commands)
    # Add starting space to utterance if it doesn't have one
    if utterance[1] != ' '
        utterance = " $utterance"
    end
    # Generate constrained trace from literal listener model
    verbose && println("Evaluating logprobs of observed utterance...")
    choices = choicemap((:utterance => :output, utterance))
    trace, _ = generate(literal_utterance_model,
                        (domain, state, commands), choices)
    # Extract unnormalized log-probabilities of utterance for each command
    command_scores = extract_utterance_scores_per_command(trace)
    # Compute posterior probability of each command
    verbose && println("Computing posterior over commands...")
    command_probs = softmax(command_scores)
    # Sort commands by posterior probability
    perm = sortperm(command_scores, rev=true)
    commands = commands[perm]
    command_probs = command_probs[perm]
    command_scores = command_scores[perm]
    # Return commands and their posterior probabilities
    return (
        commands = commands,
        probs = command_probs,
        scores = command_scores
    )
end

## Pragmatic Inference ##

"""
    configure_pragmatic_speaker_model(
        domain, state, goals, cost_profiles;
        act_temperatures = [1.0],
        modalities = (:utterance, :action),
        max_nodes = 2^16
    )

Configure the listener / assistant's model of the speaker / human principal.
"""
function configure_pragmatic_speaker_model(
    domain::Domain, state::State,
    goals::AbstractVector{<:Term},
    cost_profiles;
    act_temperatures = [1.0],
    modalities = (:utterance, :action),
    n_iters = 1,
    max_nodes = 2^18,
    kwargs...
)
    # Define goal prior
    @gen function goal_prior()
        # Sample goal index
        goal ~ uniform_discrete(1, length(goals))
        # Sample action costs
        cost_idx ~ uniform_discrete(1, length(cost_profiles))
        costs = cost_profiles[cost_idx]
        # Construct goal specification
        spec = MinPerAgentActionCosts(Term[goals[goal]], costs)
        return spec
    end

    # Configure planner
    heuristic = precomputed(DoorsKeysMSTHeuristic(), domain, state)
    planner = RTHS(heuristic=heuristic, n_iters=n_iters,
                   max_nodes=max_nodes, fail_fast=true)

    # Convert act temperature parameter
    if act_temperatures isa Real
        act_args = ([act_temperatures],)
    elseif act_temperatures isa AbstractVector
        act_args = (act_temperatures,)
    elseif act_temperatures isa Tuple && act_temperatures[1] isa Gen.Distribution
        act_args = act_temperatures
        dist, dist_args = act_args[1:2]
        temp_base = act_args[3]
        temp_exp_range = act_args[4]
        temps = temp_base .^ collect(temp_exp_range)
        act_args = (temps, dist, dist_args)
    else
        act_args = act_temperatures
    end

    # Define communication and action configuration
    act_config = HierarchicalBoltzmannActConfig(act_args...)
    if :utterance in modalities
        act_config = CommunicativeActConfig(
            act_config, # Assume some Boltzmann action noise
            pragmatic_utterance_model, # Utterance model
            (domain, planner) # Domain and planner are arguments to utterance model
        )
    end

    # Define agent configuration
    agent_config = AgentConfig(
        domain, planner;
        # Assume fixed goal over time
        goal_config = StaticGoalConfig(goal_prior),
        # Assume the agent refines its policy at every timestep
        replan_args = (
            plan_at_init = true, # Plan at initial timestep
            prob_replan = 0, # Probability of replanning at each timestep
            prob_refine = 1.0, # Probability of refining solution at each timestep
            rand_budget = false # Search budget is fixed everytime
        ),
        act_config = act_config
    )

    # Configure world model with agent and environment configuration
    world_config = WorldConfig(
        agent_config = agent_config,
        env_config = PDDLEnvConfig(domain, state),
        obs_config = PerfectObsConfig()
    )

    return world_config
end

"""
    pragmatic_goal_inference(
        model_config, n_goals, n_costs,
        actions, utterances, utterance_times;
        modalities = (:utterance, :action),
        verbose = false,
        kwargs...
    )

Runs online pragmatic goal inference problem defined by `model_config` and the 
observed `actions` and `utterances`. Returns the particle filter state, the
distribution over goals, the distribution over cost profiles, and the log
marginal likelihood estimate of the data.
"""
function pragmatic_goal_inference(
    model_config::WorldConfig,
    n_goals::Int, n_costs::Int,
    actions::AbstractVector{<:Term},
    utterances::AbstractVector{String},
    utterance_times::AbstractVector{Int};
    speaker = pddl"(human)",
    listener = pddl"(robot)",
    modalities = (:utterance, :action),
    verbose = false,
    goal_names = ["gem$i" for i in 1:n_goals],
    kwargs...
)
    # Add do-operator to listener actions (all actions for utterance-only model)
    obs_actions = map(actions) do act
        :action ∉ modalities || act.args[1] == listener ?
            InversePlanning.do_op(act) : act
    end
    # Convert plan to action choicemaps
    observations = act_choicemap_vec(obs_actions)
    timesteps = collect(1:length(observations))
    # Construct selection containing all action addresses
    action_sel = Gen.select()
    for t in timesteps
        push!(action_sel, :timestep => t => :act => :act)
    end
    # Add utterances to choicemaps
    utterance_sel = Gen.select()
    if :utterance in modalities
        # Set `speak` to false for all timesteps
        for (t, obs) in zip(timesteps, observations)
            obs[:timestep => t => :act => :speak] = false
        end
        # Add initial choice map
        init_obs = choicemap((:init => :act => :speak, false))
        pushfirst!(observations, init_obs)
        pushfirst!(timesteps, 0)
        # Constrain `speak` and `utterance` for each step where speech occurs
        for (t, utt) in zip(utterance_times, utterances)
            if utt[1] != ' ' # Add starting space to utterance if missing
                utt = " $utt"
            end
            if t == 0
                speak_addr = :init => :act => :speak
                utterance_addr = :init => :act => :utterance => :output
            else
                speak_addr = :timestep => t => :act => :speak
                utterance_addr = :timestep => t => :act => :utterance => :output
            end
            observations[t+1][speak_addr] = true
            observations[t+1][utterance_addr] = utt
            push!(utterance_sel, speak_addr)
            push!(utterance_sel, utterance_addr)
        end
    end

    # Construct iterator over goals and cost profiles for stratified sampling
    goal_addr = :init => :agent => :goal => :goal
    cost_addr = :init => :agent => :goal => :cost_idx
    init_strata = choiceproduct((goal_addr, 1:n_goals),
                                (cost_addr, 1:n_costs))

    # Construct logging and printing callbacks
    logger_cb = DataLoggerCallback(
        t = (t, pf) -> t::Int,
        goal_probs = pf -> probvec(pf, goal_addr, 1:n_goals)::Vector{Float64},
        cost_probs = pf -> probvec(pf, cost_addr, 1:n_costs)::Vector{Float64},
        lml_est = pf -> log_ml_estimate(pf)::Float64,
        action_goal_probs = pf -> begin
            tr_scores = map(pf.traces) do trace
                project(trace, action_sel)
            end
            tr_probs = softmax(tr_scores)
            probs = zeros(n_goals)
            for (idx, tr) in enumerate(pf.traces)
                goal = tr[goal_addr]
                probs[goal] += tr_probs[idx]
            end
            return probs
        end,
        utterance_goal_probs = pf -> begin
            tr_scores = map(pf.traces) do trace
                project(trace, utterance_sel)
            end
            tr_probs = softmax(tr_scores)
            probs = zeros(n_goals)
            for (idx, tr) in enumerate(pf.traces)
                goal = tr[goal_addr]
                probs[goal] += tr_probs[idx]
            end
            return probs
        end,
        trace_scores = pf -> get_log_weights(pf),
        action_trace_scores = pf -> begin
            map(tr -> project(tr, action_sel), pf.traces)
        end,
        utterance_trace_scores = pf -> begin
            map(tr -> project(tr, utterance_sel), pf.traces)
        end,
    )
    print_cb = PrintStatsCallback(
        (goal_addr, 1:n_goals),
        (cost_addr, 1:n_costs);
        header=("t\t" * join(goal_names, "\t") * "\t" *
                join(["C$C" for C in 1:n_costs], "\t"))
    )
    if verbose
        callback = CombinedCallback(logger=logger_cb, print=print_cb)
    else
        callback = CombinedCallback(logger=logger_cb)
    end

    # Configure SIPS particle filter
    sips = SIPS(model_config, resample_cond=:none, rejuv_cond=:none)
    # Run particle filter to perform online goal inference
    n_samples = length(init_strata)
    pf_state = sips(
        n_samples,  observations;
        init_args=(init_strata=init_strata,),
        callback=callback
    );

    # Extract logged data
    goal_probs_history = callback.logger.data[:goal_probs]
    goal_probs_history = reduce(hcat, goal_probs_history)
    goal_probs = goal_probs_history[:, end]
    cost_probs_history = callback.logger.data[:cost_probs]
    cost_probs_history = reduce(hcat, cost_probs_history)
    cost_probs = cost_probs_history[:, end]
    lml_est_history = callback.logger.data[:lml_est]
    lml_est = lml_est_history[end]
    action_goal_probs_history = callback.logger.data[:action_goal_probs]
    action_goal_probs_history = reduce(hcat, action_goal_probs_history)
    action_goal_probs = action_goal_probs_history[:, end]
    utterance_goal_probs_history = callback.logger.data[:utterance_goal_probs]
    utterance_goal_probs_history = reduce(hcat, utterance_goal_probs_history)
    utterance_goal_probs = utterance_goal_probs_history[:, end]
    trace_score_history = callback.logger.data[:trace_scores]
    trace_score_history = reduce(hcat, trace_score_history)
    trace_scores = trace_score_history[:, end]
    action_trace_score_history = callback.logger.data[:action_trace_scores]
    action_trace_score_history = reduce(hcat, action_trace_score_history)
    action_trace_scores = action_trace_score_history[:, end]
    utterance_trace_score_history = callback.logger.data[:utterance_trace_scores]
    utterance_trace_score_history = reduce(hcat, utterance_trace_score_history)
    utterance_trace_scores = utterance_trace_score_history[:, end]

    return (
        pf = pf_state,
        sips_config = sips,
        goal_probs = goal_probs,
        cost_probs = cost_probs,
        action_goal_probs = action_goal_probs,
        utterance_goal_probs = utterance_goal_probs,
        trace_scores = trace_scores,
        action_trace_scores = action_trace_scores,
        utterance_trace_scores = utterance_trace_scores,
        lml_est = lml_est,
        goal_probs_history = goal_probs_history,
        cost_probs_history = cost_probs_history,
        action_goal_probs_history = action_goal_probs_history,
        utterance_goal_probs_history = utterance_goal_probs_history,
        trace_score_history = trace_score_history,
        action_trace_score_history = action_trace_score_history,
        utterance_trace_score_history = utterance_trace_score_history,
        lml_est_history = lml_est_history
    )
end

function pragmatic_goal_inference(
    domain::Domain, state::State, goals::Vector{Term}, cost_profiles,
    actions::AbstractVector{<:Term},
    utterances::AbstractVector{String},
    utterance_times::AbstractVector{Int};
    kwargs...
)
    # Configure speaker model
    model_config = configure_pragmatic_speaker_model(
        domain, state, goals, cost_profiles; kwargs...
    )
    # Run goal inference
    return pragmatic_goal_inference(
        model_config, length(goals), length(cost_profiles),
        actions, utterances, utterance_times; kwargs...
    )
end
