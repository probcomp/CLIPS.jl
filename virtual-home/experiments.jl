using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using InversePlanning
using Printf
using CSV, DataFrames
using Dates

using GenParticleFilters: softmax

include("src/utils.jl")
include("src/plan_io.jl")
include("src/heuristics.jl")
include("src/utterances.jl")
include("src/inference.jl")
include("src/assistance.jl")

## Load domains, problems and plans ##

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
GOAL_DIR = joinpath(@__DIR__, "dataset", "goals", "definitions")
GOALSETS_DIR = joinpath(@__DIR__, "dataset", "goals", "sets")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans", "observed")
COMPLETION_DIR = joinpath(@__DIR__, "dataset", "plans", "completed")

RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

# Load domain
DOMAIN = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
COMPILED_DOMAINS = Dict{String, Domain}()

# Load problems
PROBLEMS = Dict{String, Problem}()
for path in readdir(PROBLEM_DIR)
    name, ext = splitext(path)
    ext == ".pddl" || continue
    PROBLEMS[name] = load_problem(joinpath(PROBLEM_DIR, path))
end

# Load goal dataset
ALL_GOAL_LABELS, GOALS = load_goal_dataset(GOAL_DIR)

# Load utterance-annotated plans and completions
PLAN_IDS, PLANS, TRUE_GOALS, UTTERANCES, UTTERANCE_TIMES = load_plan_dataset(PLAN_DIR)
_, COMPLETIONS, _, _, _ = load_plan_dataset(COMPLETION_DIR)

## Define parameters ##

# Possible cost profiles
COST_PROFILES = [nothing]

# Boltzmann action temperatures / temperature priors
ACT_TEMPERATURES = [(inv_gamma, (1.0, 2.0), 2, -3:0.25:5)]

# Possible modalities
MODALITIES = [
    (:action,),
    (:utterance,),
    (:action, :utterance),
]

# Maximum number of steps before time is up
MAX_STEPS = 50

# Number of samples for systematic sampling
N_LITERAL_SAMPLES = 10

# Whether to run literal or pragmatic inference
RUN_LITERAL = true
RUN_PRAGMATIC = true

## Run experiments ##

df = DataFrame(
    # Plan info
    plan_id = String[],
    problem_id = String[],
    assist_objs = String[],
    goal_count = Int[],
    option_count = Int[],
    utterances =  String[],
    true_goal = String[],
    true_assist_options = String[],
    # Method info
    infer_method = String[],
    assist_method = String[],
    estim_type = String[],
    modalities = String[],
    act_temperature = String[],
    # Inference results
    goal_probs_1 = Float64[],
    goal_probs_2 = Float64[],
    goal_probs_3 = Float64[],
    goal_probs_4 = Float64[],
    goal_probs_5 = Float64[],
    goal_probs_6 = Float64[],
    goal_probs_7 = Float64[],
    goal_probs_8 = Float64[],
    goal_probs_9 = Float64[],
    goal_probs_10 = Float64[],
    goal_probs_11 = Float64[],
    goal_probs_12 = Float64[],
    true_goal_probs = Float64[],
    brier_score = Float64[],
    lml_est = Float64[],
    # Assistance results
    top_command = String[],
    top_command_prob = Float64[],
    top_5_commands = String[],
    top_5_command_probs = String[],
    assist_probs_1 = Float64[],
    assist_probs_2 = Float64[],
    assist_probs_3 = Float64[],
    assist_probs_4 = Float64[],
    assist_probs_5 = Float64[],
    assist_probs_6 = Float64[],
    assist_probs_7 = Float64[],
    assist_probs_8 = Float64[],
    assist_probs_9 = Float64[],
    assist_probs_10 = Float64[],
    assist_probs_11 = Float64[],
    assist_probs_12 = Float64[],
    assist_probs_13 = Float64[],
    assist_probs_14 = Float64[],
    assist_probs_15 = Float64[],
    assist_probs_16 = Float64[],    
    assist_precision = Float64[],
    assist_recall = Float64[],
    cmd_success = Float64[],
    goal_success = Float64[],
    assist_plan = String[],
    plan_length = Float64[],
    plan_cost = Float64[],
    speaker_cost = Float64[]
)
datetime = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
df_types = eltype.(eachcol(df))
df_path = "experiments_$(datetime).csv"
df_path = joinpath(RESULTS_DIR, df_path)

inference_df = DataFrame(
    # Plan info
    plan_id = String[],
    problem_id = String[],
    true_goal = String[],
    # Method info
    act_temperature = String[],
    # Inference results
    timestep = Int64[],
    lml_est = Float64[];
    (Symbol("goal_probs_$(i)") => Float64[] for i in 1:12)...,
    (Symbol("act_goal_probs_$(i)") => Float64[] for i in 1:12)...,
    (Symbol("utt_goal_probs_$(i)") => Float64[] for i in 1:12)...,
    (Symbol("cost_probs_$(i)") => Float64[] for i in 1:12)...,
    (Symbol("trace_scores_$(i)") => Float64[] for i in 1:12)...,
    (Symbol("act_trace_scores_$(i)") => Float64[] for i in 1:12)...,
    (Symbol("utt_trace_scores_$(i)") => Float64[] for i in 1:12)...
)
inference_df_path = "inferences_per_timestep_$(datetime).csv"
inference_df_path = joinpath(RESULTS_DIR, inference_df_path)

# Iterate over plans
for plan_id in PLAN_IDS
    println("=== Plan $plan_id ===")
    # Load plan and problem
    plan = PLANS[plan_id]
    true_goal_label = TRUE_GOALS[plan_id]
    utterances = UTTERANCES[plan_id]
    utterance_times = UTTERANCE_TIMES[plan_id]
    completion = COMPLETIONS[plan_id]
    println(utterances)
    
    problem_id = match(r"(\d+).(\d+)", plan_id).captures[1]
    problem = PROBLEMS[problem_id]
    
    # Load set of possible goals
    goal_labels = readlines(joinpath(GOALSETS_DIR, problem_id * ".txt"))
    goals = [GOALS[label] for label in goal_labels]
    
    # Load true goal
    true_goal_label = TRUE_GOALS[plan_id]
    true_goal = GOALS[true_goal_label]
    
    # Construct true goal specification
    true_goal_spec = MinStepsGoal(true_goal)
    cost_profiles = COST_PROFILES

    # Compile domain for problem
    domain = get!(COMPILED_DOMAINS, problem_id) do
        println("Compiling domain for problem $problem_id...")
        state = initstate(DOMAIN, problem)
        domain, _ = PDDL.compiled(DOMAIN, state)
        return domain
    end

    # Construct initial state
    state = initstate(domain, problem)
    # Simulate plan to completion
    plan_end_state = EndStateSimulator()(domain, state, plan)
    remain_steps = MAX_STEPS - length(plan)

    # Extract true assistance options from completion
    true_assist_options, assist_objs =
        extract_assist_options(plan_end_state, completion)

    # Construct plan entry for dataframe
    plan_entry = Dict{Symbol, Any}(
        :plan_id => plan_id,
        :problem_id => problem_id,
        :assist_objs => join(string.(assist_objs), ", "),
        :goal_count => length(goals),
        :option_count => length(assist_objs),
        :true_goal => true_goal_label,
        :true_assist_options => join(string.(true_assist_options), ", "),
    )

    # Run literal inference and assistance
    if RUN_LITERAL
        println()
        println("-- Literal instruction following --")

        # Infer distribution over commands
        commands, command_probs, command_scores =
            literal_command_inference(domain, plan_end_state, utterances,
                                      verbose=true)
        top_command = commands[1]

        # Print top 5 commands and their probabilities
        println("Top 5 most probable commands:")
        for idx in 1:5
            command_str = repr("text/plain", commands[idx])
            @printf("%.3f: %s\n", command_probs[idx], command_str)
        end

        # Set up planners
        heuristic = precomputed(VirtualHomeHeuristic(), domain, plan_end_state)
        cmd_planner = AStarPlanner(heuristic, max_nodes=2^16,
                                   fail_fast=true, verbose=true)
        goal_planner = AStarPlanner(heuristic, max_nodes=2^16,
                                    fail_fast=true, verbose=true)

        # Set up dataframe entry
        entry = copy(plan_entry)
        entry[:utterances] = join(utterances, "\n")
        entry[:infer_method] = "literal"
        entry[:assist_method] = "literal"
        entry[:modalities] = "utterance"
        entry[:act_temperature] = string(0.0)
        entry[:top_command] = repr("text/llm", top_command)
        entry[:top_command_prob] = command_probs[1]
        entry[:top_5_commands] =
            join([repr("text/llm", c) for c in commands[1:5]], "\n")
        entry[:top_5_command_probs] = string(command_probs[1:5])
        for i in 1:16
            entry[Symbol("assist_probs_$i")] = 0.0
        end

        # Compute assistance options and plans for top command
        println()
        println("- Literal assistance (top command) -")
        top_assist_results = literal_assistance(
            top_command, domain, plan_end_state, true_goal_spec;
            true_assist_options, assist_objs, cmd_planner, goal_planner,
            max_steps = remain_steps, verbose = true
        )
        entry[:estim_type] = "mode"
        entry[:assist_plan] =
            join(write_pddl.(top_assist_results.full_plan), "\n")
        entry[:plan_length] = top_assist_results.plan_length
        entry[:plan_cost] = top_assist_results.plan_cost
        entry[:speaker_cost] = top_assist_results.speaker_cost
        entry[:cmd_success] = top_assist_results.cmd_success
        entry[:goal_success] = top_assist_results.goal_success
        entry[:assist_precision] = top_assist_results.assist_precision
        entry[:assist_recall] = top_assist_results.assist_recall
        for (i, p) in enumerate(top_assist_results.assist_option_probs)
            entry[Symbol("assist_probs_$i")] = p
        end
        push!(df, entry, cols=:union)

        # Compute expected assistance options and plans via systematic sampling
        println()
        println("- Literal assistance (full distribution) -")
        mean_assist_results = literal_assistance(
            commands, command_probs,
            domain, plan_end_state, true_goal_spec;
            true_assist_options, assist_objs, cmd_planner, goal_planner,
            max_steps = remain_steps, verbose = true,
            n_samples = N_LITERAL_SAMPLES
        )
        entry[:estim_type] = "mean"
        entry[:assist_plan] = ""
        entry[:plan_length] = mean_assist_results.plan_length
        entry[:plan_cost] = mean_assist_results.plan_cost
        entry[:speaker_cost] = mean_assist_results.speaker_cost
        entry[:cmd_success] = mean_assist_results.cmd_success
        entry[:goal_success] = mean_assist_results.goal_success
        entry[:assist_precision] = mean_assist_results.assist_precision
        entry[:assist_recall] = mean_assist_results.assist_recall
        for (i, p) in enumerate(mean_assist_results.assist_option_probs)
            entry[Symbol("assist_probs_$i")] = p
        end
        push!(df, entry, cols=:union)

        GC.gc()
        CSV.write(df_path, df)
    end

    # Run pragmatic inference and assistance
    if RUN_PRAGMATIC
        println()
        println("-- Pragmatic instruction following --")

        # Set up dataframe entry        
        entry = copy(plan_entry)
        entry[:utterances] = join(utterances, "\n")
        entry[:infer_method] = "pragmatic"
        entry[:top_command] = ""
        for i in 1:16
            entry[Symbol("assist_probs_$i")] = 0.0
        end

        # Iterate over modalities and parameters
        for act_temperature in ACT_TEMPERATURES
            println()
            println("Action temperature: $act_temperature")
            entry[:act_temperature] = string(act_temperature)

            # Configure pragmatic speaker/agent model
            model_config = configure_pragmatic_speaker_model(
                domain, state, goals, cost_profiles;
                modalities=(:action, :utterance),
                act_temperatures = act_temperature
            )

            # Run goal inference
            println()
            println("Running pragmatic goal inference...")
            pragmatic_inference_results = pragmatic_goal_inference(
                model_config, length(goals), length(cost_profiles),
                plan, utterances, utterance_times,
                verbose = true, goal_labels = goal_labels
            )
            sips_config = pragmatic_inference_results.sips_config

            # Store inference results per timestep
            rs = pragmatic_inference_results
            n_steps = length(plan) + 1
            n_goals = length(goals)
            n_traces = length(goals) * length(cost_profiles)
            new_inference_df = DataFrame(
                plan_id = fill(plan_id, n_steps),
                problem_id = fill(problem_id, n_steps),
                true_goal = fill(true_goal_label, n_steps),
                act_temperature = fill(act_temperature, n_steps),
                timestep = 0:(n_steps-1),
                lml_est = rs.lml_est_history;
                (Symbol("goal_probs_$(i)") => rs.goal_probs_history[i, :] for i in 1:n_goals)...,
                (Symbol("act_goal_probs_$(i)") => rs.action_goal_probs_history[i, :] for i in 1:n_goals)...,
                (Symbol("utt_goal_probs_$(i)") => rs.utterance_goal_probs_history[i, :] for i in 1:n_goals)...,
                (Symbol("cost_probs_$(i)") => rs.cost_probs_history[i, :] for i in 1:length(cost_profiles))...,
                (Symbol("trace_scores_$(i)") => rs.trace_score_history[i, :] for i in 1:n_traces)...,
                (Symbol("act_trace_scores_$(i)") => rs.action_trace_score_history[i, :] for i in 1:n_traces)...,
                (Symbol("utt_trace_scores_$(i)") => rs.utterance_trace_score_history[i, :] for i in 1:n_traces)...
            )
            append!(inference_df, new_inference_df, cols=:union)
            CSV.write(inference_df_path, inference_df)

            for modalities in MODALITIES
                println()
                println("Modalities: $modalities")
                entry[:modalities] = join(collect(modalities), ", ")

                # Store inference results for modality
                goal_probs = if modalities == (:action,)
                    pragmatic_inference_results.action_goal_probs
                elseif modalities == (:utterance,)
                    pragmatic_inference_results.utterance_goal_probs
                elseif modalities == (:action, :utterance)
                    pragmatic_inference_results.goal_probs
                end
                for (i, p) in enumerate(goal_probs)
                    entry[Symbol("goal_probs_$i")] = p
                end
                true_goal_idx = findfirst(==(true_goal), goals)
                entry[:true_goal_probs] = goal_probs[true_goal_idx]
                entry[:brier_score] =
                    sum((goal_probs .- (1:length(goals) .== true_goal_idx)).^2)
                entry[:lml_est] = pragmatic_inference_results.lml_est

                # Run assistive policies
                println("Running pragmatic goal assistance...")
                pf = copy(pragmatic_inference_results.pf)
                pf.log_weights = copy(pf.log_weights)
                if modalities == (:action,)
                    pf.log_weights .=
                        pragmatic_inference_results.action_trace_scores
                elseif modalities == (:utterance,)
                    pf.log_weights .=
                        pragmatic_inference_results.utterance_trace_scores
                end

                # Assistance by following policy to most likely goal
                println("- Pragmatic assistance π*-MDP (MAP policy) -")
                pragmatic_pmdp_mode_results = pragmatic_assistance_pmdp(
                    pf, domain, plan_end_state, true_goal_spec;
                    true_plan = completion, true_assist_options, assist_objs,
                    max_steps = MAX_STEPS, verbose = true,
                    estim_type = "mode"
                )
                println()

                entry[:assist_method] = "pmdp"
                entry[:estim_type] = "mode"
                entry[:assist_plan] =
                    join(write_pddl.(pragmatic_pmdp_mode_results.full_plan), "\n")
                entry[:plan_length] = pragmatic_pmdp_mode_results.plan_length
                entry[:plan_cost] = pragmatic_pmdp_mode_results.plan_cost
                entry[:speaker_cost] = pragmatic_pmdp_mode_results.speaker_cost
                entry[:goal_success] = pragmatic_pmdp_mode_results.goal_success
                entry[:assist_precision] = pragmatic_pmdp_mode_results.assist_precision
                entry[:assist_recall] = pragmatic_pmdp_mode_results.assist_recall
                for (i, p) in enumerate(pragmatic_pmdp_mode_results.assist_option_probs)
                    entry[Symbol("assist_probs_$i")] = p
                end
                push!(df, entry, cols=:union)

                # Assistance by sampling a goal and following the corresponding policy
                println("- Pragmatic assistance ̄π-MDP (full goal distribution) -")
                pragmatic_pmdp_mean_results = pragmatic_assistance_pmdp(
                    pf, domain, plan_end_state, true_goal_spec;
                    true_plan = completion, true_assist_options, assist_objs,
                    max_steps = MAX_STEPS, verbose = true,
                    estim_type = "mean"
                )
                println()

                entry[:assist_method] = "pmdp"
                entry[:estim_type] = "mean"
                entry[:assist_plan] =
                    join(write_pddl.(pragmatic_pmdp_mean_results.full_plan), "\n")
                entry[:plan_length] = pragmatic_pmdp_mean_results.plan_length
                entry[:plan_cost] = pragmatic_pmdp_mean_results.plan_cost
                entry[:speaker_cost] = pragmatic_pmdp_mean_results.speaker_cost
                entry[:goal_success] = pragmatic_pmdp_mean_results.goal_success
                entry[:assist_precision] = pragmatic_pmdp_mean_results.assist_precision
                entry[:assist_recall] = pragmatic_pmdp_mean_results.assist_recall
                for (i, p) in enumerate(pragmatic_pmdp_mean_results.assist_option_probs)
                    entry[Symbol("assist_probs_$i")] = p
                end
                push!(df, entry, cols=:union)

                # Assistance via expected cost minimization over actions
                println("- Pragmatic assistance Q-MDP (act-level expected cost minimization) -")
                pragmatic_qmdp_act_results = pragmatic_assistance_qmdp_act(
                    pf, domain, plan_end_state, true_goal_spec;
                    true_plan = completion, true_assist_options, assist_objs,
                    max_steps = MAX_STEPS, verbose = true
                )
                println()

                entry[:assist_method] = "qmdp_act"
                entry[:estim_type] = "mode"
                entry[:assist_plan] =
                    join(write_pddl.(pragmatic_qmdp_act_results.full_plan), "\n")
                entry[:plan_length] = pragmatic_qmdp_act_results.plan_length
                entry[:plan_cost] = pragmatic_qmdp_act_results.plan_cost
                entry[:speaker_cost] = pragmatic_qmdp_act_results.speaker_cost
                entry[:goal_success] = pragmatic_qmdp_act_results.goal_success
                entry[:assist_precision] = pragmatic_qmdp_act_results.assist_precision
                entry[:assist_recall] = pragmatic_qmdp_act_results.assist_recall
                for (i, p) in enumerate(pragmatic_qmdp_act_results.assist_option_probs)
                    entry[Symbol("assist_probs_$i")] = p
                end
                push!(df, entry, cols=:union)

                # Assistance via expected cost minimization with belief updating
                println("- Pragmatic assistance Q-MDP (act-level, with belief-updating) -")
                pragmatic_qmdp_interactive_results = pragmatic_assistance_qmdp_act(
                    pf, domain, plan_end_state, true_goal_spec;
                    true_plan = completion, true_assist_options, assist_objs,
                    max_steps = MAX_STEPS, verbose = true,
                    update_beliefs = true, sips_config = sips_config
                )

                entry[:assist_method] = "qmdp_interactive"
                entry[:estim_type] = "mode"
                entry[:assist_plan] =
                    join(write_pddl.(pragmatic_qmdp_interactive_results.full_plan), "\n")
                entry[:plan_length] = pragmatic_qmdp_interactive_results.plan_length
                entry[:plan_cost] = pragmatic_qmdp_interactive_results.plan_cost
                entry[:speaker_cost] = pragmatic_qmdp_interactive_results.speaker_cost
                entry[:goal_success] = pragmatic_qmdp_interactive_results.goal_success
                entry[:assist_precision] = pragmatic_qmdp_interactive_results.assist_precision
                entry[:assist_recall] = pragmatic_qmdp_interactive_results.assist_recall
                for (i, p) in enumerate(pragmatic_qmdp_interactive_results.assist_option_probs)
                    entry[Symbol("assist_probs_$i")] = p
                end
                push!(df, entry, cols=:union)
                
                CSV.write(df_path, df)
            end
            GC.gc()
        end
    end
    println()
end
