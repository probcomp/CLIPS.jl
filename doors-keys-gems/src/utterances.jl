using PDDL, SymbolicPlanners
using Gen, GenGPT3
using IterTools
using Random
using StatsBase

include("utils.jl")
include("commands.jl")

# Example translations from salient actions to instructions/requests
pragmatic_utterance_examples = [
    # Single assistant actions (no predicates)
    ("(pickup you key1)",
     "Get the key over there."),
    ("(handover you me key1)",
     "Can you hand me that key?"),
    ("(unlock you key1 door1)",
     "Unlock this door for me please."),
    # Single assistant actions (with predicates)
    ("(pickup you key1) where (iscolor key1 yellow)",
     "Please pick up the yellow key."),
    ("(handover you me key1) where (iscolor key1 blue)",
     "Could you pass me the blue key?"),
    ("(unlock you key1 door1) where (iscolor door1 blue)",
     "Can you open the blue door?"),
    # Multiple assistant actions (distinct)
    ("(unlock you key1 door1) (handover you me key2)",
     "Would you unlock the door and bring me that key?"),
    ("(handover you me key1) (handover you me key2) where (iscolor key1 green) (iscolor key2 red)",
     "Hand me the green and red keys."),
    ("(unlock you key1 door1) (unlock you key2 door2) where (iscolor door1 green) (iscolor door2 yellow)",
     "Help me unlock the green and yellow doors."),
    # Multiple assistant actions (combined)
    ("(pickup you key1) (pickup you key2) where (iscolor key1 green) (iscolor key2 green)",
     "Can you go and get the green keys?"),
    ("(handover you me key1) (handover you me key2) where (iscolor key1 red) (iscolor key2 red)",
     "Can you pass me two red keys?"),
    ("(unlock you key1 door1) (unlock you key2 door2) (unlock you key3 door3)",
     "Could you unlock these three doors for me?"),
    # Joint actions (all distinct)
    ("(pickup me key1) (pickup you key2) where (iscolor key1 red) (iscolor key2 blue)",
     "I will get the red key, can you pick up the blue one?"),
    ("(unlock you key1 door1) (pickup me key2) where (iscolor door1 green) (iscolor key2 blue)",
     "I'm getting the blue key, can you open the green door?"),
    ("(pickup you key1) (pickup me key2) where (iscolor key1 yellow) (iscolor key2 blue)",
     "Can you pick up the yellow key while I get the blue one?"),
    # Joint actions (some combined)
    ("(pickup me key1) (handover you me key2) (handover you me key3) where (iscolor key1 blue) (iscolor key2 yellow) (iscolor key3 yellow)",
     "Can you hand me the yellow keys? I'm getting the blue one."),
    ("(pickup me key1) (pickup me key2) (unlock you key3 door1) (unlock you key4 door2)",
     "I'm picking up these keys, can you unlock those doors?"),
    ("(handover you me key1) (handover you me key2) (pickup me ?gem1) where (iscolor key1 red) (iscolor key2 red) (iscolor gem1 green)",
     "Pass me the red keys, I'm going for the green gem.")
]

# Exclude utterances about joint actions for literal listener prompt
literal_utterance_examples = pragmatic_utterance_examples[1:12]

# Shuffle the examples
rng = Random.MersenneTwister(0)
pragmatic_utterance_examples_s = shuffle(rng, pragmatic_utterance_examples)
literal_utterance_examples_s = shuffle(rng, literal_utterance_examples)

"Extract salient actions (with predicate modifiers) from a plan."
function extract_salient_actions(
    domain::Domain, state::State, plan::AbstractVector{<:Term};
    salient_actions = [
        (:pickup, 1, 2),
        (:handover, 1, 3),
        (:unlock, 1, 3)
    ],
    salient_predicates = [
        (:iscolor, (d, s, o) -> get_obj_color(s, o))
    ]
)
    actions = Term[]
    agents = Const[]
    predicates = Vector{Term}[]
    for act in plan # Extract manually-defined salient actions
        for (name, agent_idx, obj_idx) in salient_actions
            if act.name == name
                agent = act.args[agent_idx]
                obj = act.args[obj_idx]
                push!(actions, act)
                push!(agents, agent)
                push!(predicates, Term[])
                for (pred_name, pred_fn) in salient_predicates
                    val = pred_fn(domain, state, obj)
                    if val != pddl"(none)"
                        pred = Compound(pred_name, Term[obj, val])
                        push!(predicates[end], pred)
                    end
                end
            end
        end
    end
    return actions, agents, predicates
end

"Enumerate all possible salient actions (with predicate modifiers) in a state."
function enumerate_salient_actions(
    domain::Domain, state::State;
    salient_actions = [
        (:pickup, 1, 2),
        (:handover, 1, 3),
        (:unlock, 1, 3)
    ],
    salient_agents = [
        pddl"(robot)"
    ],
    salient_predicates = [
        (:iscolor, (d, s, o) -> get_obj_color(s, o))
    ]
)
    actions = Term[]
    agents = Const[]
    predicates = Vector{Term}[]
    statics = PDDL.infer_static_fluents(domain)
    # Enumerate over salient actions
    for (name, agent_idx, obj_idx) in salient_actions
        act_schema = PDDL.get_action(domain, name)
        # Enumerate over all possible groundings
        args_iter = PDDL.groundargs(domain, state, act_schema; statics)
        for args in args_iter
            # Skip actions with non-salient agents
            agent = args[agent_idx]
            agent in salient_agents || continue
            # Construct action term
            act = Compound(name, collect(Term, args))
            # Substitute and simplify precondition
            act_vars = PDDL.get_argvars(act_schema)
            subst = PDDL.Subst(var => val for (var, val) in zip(act_vars, args))
            precond = PDDL.substitute(PDDL.get_precond(act_schema), subst)
            precond = PDDL.simplify_statics(precond, domain, state, statics)
            # Skip actions that are never possible
            precond.name == false && continue
            # Add action and agent
            push!(actions, act)
            push!(agents, agent)
            # Extract predicates
            push!(predicates, Term[])
            obj = act.args[obj_idx]
            for (pred_name, pred_fn) in salient_predicates
                val = pred_fn(domain, state, obj)
                if val != pddl"(none)"
                    pred = Compound(pred_name, Term[obj, val])
                    push!(predicates[end], pred)
                end
            end
        end
    end
    return actions, agents, predicates
end

"Enumerate action commands from salient actions and predicates."
function enumerate_commands(
    actions::Vector{Term},
    agents::Vector{Const},
    predicates::Vector{Vector{Term}};
    speaker = pddl"(human)",
    listener = pddl"(robot)",
    max_commanded_actions = 3,
    max_distinct_actions = 2,
    exclude_unpredicated = true,
    exclude_predicated = false,
    exclude_action_chains = false,
    exclude_speaker_commands = true,
    exclude_speaker_only_commands = true
)
    commands = ActionCommand[]
    # Filter out speaker actions
    if exclude_speaker_commands
        idxs = filter(i -> agents[i] != speaker, 1:length(agents))
        actions = actions[idxs]
        agents = agents[idxs]
        predicates = predicates[idxs]
    end
    # Replace speaker and listener names in actions
    actions = map(actions) do act
        names_to_pronouns(act; speaker, listener)
    end
    # Enumerate commands of increasing length
    max_commanded_actions = min(max_commanded_actions, length(actions))
    for n in 1:max_commanded_actions
        # Iterate over subsets of planned actions
        for idxs in IterTools.subsets(1:length(actions), n)
            # Skip subsets where all actions are speaker actions
            if exclude_speaker_only_commands
                if all(a == speaker for a in @view(agents[idxs])) continue end
            end
            # Skip subsets with too many distinct actions
            if n > max_distinct_actions
                agent_act_pairs = [(agents[i], actions[i].name) for i in idxs]
                agent_act_pairs = unique!(agent_act_pairs)
                n_distinct_actions = length(agent_act_pairs)
                if n_distinct_actions > max_distinct_actions continue end
            end
            # Skip subsets where future actions depend on previous ones
            if exclude_action_chains
                skip = false
                objects = Set{Const}()
                for act in @view(actions[idxs]), arg in act.args
                    (arg == pddl"(you)" || arg == pddl"(me)") && continue
                    if arg in objects
                        skip = true
                        break
                    end
                    push!(objects, arg)
                end
                skip && continue
            end
            # Add command without predicate modifiers
            if !exclude_unpredicated
                cmd = ActionCommand(actions[idxs], Term[])
                push!(commands, cmd)
            end
            if exclude_predicated
                continue
            end
            # Skip subsets with too many distinct predicate modifiers
            if n > max_distinct_actions
                action_groups = [Int[] for _ in agent_act_pairs]
                for i in idxs
                    agent, act = agents[i], actions[i]
                    idx = findfirst(p -> p[1] == agent && p[2] == act.name,
                                    agent_act_pairs)
                    push!(action_groups[idx], i)
                end
                skip = false
                for group in action_groups
                    length(group) > 1 || continue
                    ref_predicates = map(predicates[group[1]]) do pred
                        obj, val = pred.args
                        return Compound(pred.name, Term[Var(:X), val])
                    end
                    for i in group[2:end], pred in predicates[i]
                        obj, val = pred.args
                        lifted_pred = Compound(pred.name, Term[Var(:X), val])
                        if lifted_pred ∉ ref_predicates
                            skip = true
                            break
                        end
                    end
                    skip && break
                end
                skip && continue
            end
            # Add command with predicate modifiers
            preds = reduce(vcat, @view(predicates[idxs]))
            cmd = ActionCommand(actions[idxs], preds)
            push!(commands, cmd)
        end
    end
    return commands
end

"Construct utterance prompt from an action command and previous examples."
function construct_utterance_prompt(command::ActionCommand, examples)
    # Empty prompt if nothing to communicate
    if isempty(command.actions) return "\n" end
    # Construct example string
    example_strs = ["Input: $cmd\nOutput: $utt" for (cmd, utt) in examples]
    example_str = join(example_strs, "\n")
    command_str = repr(MIME"text/llm"(), command)
    prompt = "$example_str\nInput: $command_str\nOutput:"
    return prompt
end

"Extract unnormalized logprobs of utterance conditioned on each command."
function extract_utterance_scores_per_command(trace::Trace, addr=:utterance)
    # Extract GPT-3 mixture trace over utterances
    utt_trace = trace.trie.leaf_nodes[addr].subtrace_or_retval
    return utt_trace.scores
end

"Extract command distribution from pragmatic goal inference trace."
function extract_inferred_commands(trace::Trace, t::Int)
    if t == 0
        step_trace = Gen.static_get_subtrace(trace, Val(:init))
    else
        step_trace = Gen.static_get_subtrace(trace, Val(:timestep)).subtraces[t]
    end
    act_trace = step_trace.trie[:act].subtrace_or_retval
    utt_trace = act_trace.trie[:utterance].subtrace_or_retval
    prompts = utt_trace.prompts
    commands = map(prompts) do p
        isempty(p) ? "" : split(p, "\n")[end-1]
    end
    scores = utt_trace.scores
    perm = sortperm(scores, rev=true)
    commands = commands[perm]
    scores = scores[perm]
    probs = softmax(scores)
    return commands, scores, probs
end

# Define GPT-3 mixture generative function for literal utterance model
literal_gpt3_mixture =
    GPT3Mixture(model="text-davinci-002", stop="\n", max_tokens=64)

"Literal utterance model for human instructions using an LLM likelihood."
@gen function literal_utterance_model(
    domain::Domain, state::State,
    commands = nothing,
    examples = literal_utterance_examples_s
)
    # Enumerate commands if not already provided
    if isnothing(commands)
        actions, agents, predicates = enumerate_salient_actions(domain, state)
        commands = enumerate_commands(actions, agents, predicates)
        commands = [lift_command(cmd, state) for cmd in commands]
        unique!(commands)
    end
    # Construct prompts for each action command
    if isempty(commands)
        prompts = ["\n"]
    else
        prompts = map(commands) do command
            construct_utterance_prompt(command, examples)
        end
    end
    # Sample utterance from GPT-3 mixture over prompts
    utterance ~ literal_gpt3_mixture(prompts)
    return utterance
end

# Define GPT-3 mixture generative function for pragmatic utterance model
pragmatic_gpt3_mixture =
    GPT3Mixture(model="curie", stop="\n", max_tokens=64)

"Pragmatic utterance model for human instructions using an LLM likelihood."
@gen function pragmatic_utterance_model(
    t, act_state, agent_state, env_state, act,
    domain,
    planner,
    p_speak = 0.05,
    examples = pragmatic_utterance_examples_s
)
    # Decide whether utterance should be communicated
    speak ~ bernoulli(p_speak)
    # Return empty utterance if not speaking
    !speak && return ""
    # Extract environment state, plan state and goal specification
    state = env_state
    sol = agent_state.plan_state.sol
    spec = convert(Specification, agent_state.goal_state)
    # Rollout planning solution to get future plan
    plan = rollout_sol(domain, planner, state, sol, spec)
    # Extract salient actions and predicates from plan
    actions, agents, predicates = extract_salient_actions(domain, state, plan)
    # Enumerate action commands
    commands = enumerate_commands(actions, agents, predicates)
    commands = [lift_command(cmd, state) for cmd in commands]
    unique!(commands)
    # Construct prompts for each unique action command
    if isempty(commands)
        prompts = ["\n"]
        probs = [1.0]
    else
        command_probs = proportionmap(commands) # Compute probabilities
        prompts = String[]
        probs = Float64[]
        for (command, prob) in command_probs
            prompt = construct_utterance_prompt(command, examples)
            push!(prompts, prompt)
            push!(probs, prob)
        end
    end
    # Sample utterance from GPT-3 mixture over prompts
    utterance ~ pragmatic_gpt3_mixture(prompts, probs)
    return utterance
end
