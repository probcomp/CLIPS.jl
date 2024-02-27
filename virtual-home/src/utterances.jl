using PDDL, SymbolicPlanners
using Gen, GenGPT3
using IterTools
using Random
using StatsBase

include("utils.jl")
include("commands.jl")

# Example translations from salient actions to instructions/requests
utterance_examples = [
    # Single assistant actions
    ("(grab you plate1)",
     "Can you get a plate?"),
    ("(grab you bowl1)",
     "Can you go get a bowl?"),
    ("(grab you cutleryfork1)",
     "We need a fork."),
    ("(grab you cupcake1)",
     "Get me some cupcake?"),
    # Multiple assistant actions (distinct)
    ("(grab you cutleryfork1) (grab you cutleryknife1)",
     "Can you get me a fork and knife?"),
    ("(grab you plate1) (grab you bowl1) (grab you cutleryfork1)",
     "Grab me a plate, bowl, and a fork."),
    ("(grab you carrot1) (grab you onion1)",
     "Hand me the veggies."),
    ("(grab you juice1) (grab you waterglass1)",
     "Give me juice and a glass."),
    # Multiple assistant actions (combined)
    ("(grab you cutleryfork1) (grab you cutleryfork2) (grab you wine1)",
     "Go get two forks and some wine."),
    ("(grab you plate1) (grab you plate2)",
     "Go find two plates."),
    ("(grab you plate1) (grab you plate2) (grab you bowl1) (grab you bowl2)",
     "Can you get some plates and bowls?"),
    ("(grab you waterglass1) (grab you waterglass2) (grab you waterglass3)",
     "Could you get the glasses?"),
]

# Shuffle the examples
rng = Random.MersenneTwister(0)
utterance_examples_s = shuffle(rng, utterance_examples)

"Extract salient actions (with predicate modifiers) from a plan."
function extract_salient_actions(
    domain::Domain, state::State, plan::AbstractVector{<:Term};
    salient_actions = [(:grab, 1, 2)],
    salient_predicates = [(:itemtype, (d, s, o) -> get_item_type(s, o))]
)
    actions = Term[]
    agents = Const[]
    predicates = Vector{Term}[]
    for act in plan # Extract manually-defined salient actions
        for (name, agent_idx, obj_idx) in salient_actions
            if act.name == name
                push!(actions, act)
                push!(agents, act.args[agent_idx])
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
    end
    return actions, agents, predicates
end

"Enumerate all possible salient actions (with predicate modifiers) in a state."
function enumerate_salient_actions(
    domain::Domain, state::State;
    salient_actions = [(:grab, 1, 2)],
    salient_agents = [pddl"(helper)"],
    salient_predicates = [(:itemtype, (d, s, o) -> get_item_type(s, o))],
    statics = [:itemtype, :placed]
)
    actions = Term[]
    agents = Const[]
    predicates = Vector{Term}[]
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
    speaker = pddl"(actor)",
    listener = pddl"(helper)",
    max_commanded_actions = 4,
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
            if !exclude_speaker_commands && exclude_speaker_only_commands
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
                objects = Set{Term}()
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
                            if length(ref_predicates) == max_distinct_actions
                                skip = true
                                break
                            else
                                push!(ref_predicates, lifted_pred)
                            end
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
    GPT3Mixture(model="davinci-002", stop="\n", max_tokens=64, temperature=0.5)

"Literal utterance model for actor instructions using an LLM likelihood."
@gen function literal_utterance_model(
    domain::Domain, state::State,
    commands = nothing,
    examples = utterance_examples_s
)
    # Enumerate commands if not already provided
    if isnothing(commands)
        actions, agents, predicates = enumerate_salient_actions(domain, state)
        commands = enumerate_commands(actions, agents, predicates)
        commands = lift_command.(commands, [state])
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

# Define GPT-3 mixture generative function for literal utterance model
pragmatic_gpt3_mixture =
    GPT3Mixture(model="davinci-002", stop="\n", max_tokens=64)

"Pragmatic utterance model for actor instructions using an LLM likelihood."
@gen function pragmatic_utterance_model(
    t, act_state, agent_state, env_state, act,
    domain,
    planner,
    p_speak = 0.05,
    examples = utterance_examples_s,
    single_command = false
)
    # Decide whether utterance should be communicated
    speak ~ bernoulli(p_speak)
    # Return empty utterance if not speaking
    !speak && return ""
    # Extract environment state, plann state and goal specification
    state = env_state
    sol = agent_state.plan_state.sol
    spec = convert(Specification, agent_state.goal_state)
    # Rollout planning solution to get future plan
    cmd_state = copy(state)
    cmd_state[pddl"(nograb actor)"] = true
    plan = rollout_sol(domain, planner, cmd_state, sol, spec)
    # Extract salient actions and predicates from plan
    actions, agents, predicates = extract_salient_actions(domain, state, plan)
    # Enumerate action commands
    if single_command
        predicates = isempty(predicates) ? Term[] : reduce(vcat, predicates)
        command = ActionCommand(actions, predicates)
        commands = [lift_command(command, state)]
    else
        commands = enumerate_commands(actions, agents, predicates)
        commands = [lift_command(cmd, state) for cmd in commands]
        unique!(commands)
    end
    # Construct prompts for each unique action command
    if isempty(commands)
        prompts = ["\n"]
        probs = [1.0]
    else
        prompts = [construct_utterance_prompt(cmd, examples) for cmd in commands]
        probs = ones(Float64, length(commands)) ./ length(commands)
    end
    # Sample utterance from GPT-3 mixture over prompts
    utterance ~ pragmatic_gpt3_mixture(prompts, probs)
    return utterance
end
