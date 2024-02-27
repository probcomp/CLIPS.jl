using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using InversePlanning
using Printf

using GenParticleFilters: softmax

include("src/utils.jl")
include("src/plan_io.jl")
include("src/heuristics.jl")
include("src/inference.jl")
include("src/assistance.jl")

## Load domains, problems and plans ##

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
GOAL_DIR = joinpath(@__DIR__, "dataset", "goals", "definitions")
GOALSETS_DIR = joinpath(@__DIR__, "dataset", "goals", "sets")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans", "observed")
COMPLETION_DIR = joinpath(@__DIR__, "dataset", "plans", "completed")

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

## Set-up for specific plan and problem ##

# Select plan and problem
plan_id = "2.5"

plan = PLANS[plan_id]
true_goal_label = TRUE_GOALS[plan_id]
utterances = UTTERANCES[plan_id]
utterance_times = UTTERANCE_TIMES[plan_id]
completion = COMPLETIONS[plan_id]

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

# Compile domain for problem
domain = get!(COMPILED_DOMAINS, problem_id) do
    state = initstate(DOMAIN, problem)
    domain, _ = PDDL.compiled(DOMAIN, state)
    return domain
end

# Construct initial state
state = initstate(domain, problem)

# Simulate plan to completion
plan_end_state = EndStateSimulator()(domain, state, plan)

# Extract true assistance options from completion
true_assist_options, assist_objs =
    extract_assist_options(plan_end_state, completion)

# Test planning to true goal
heuristic = precomputed(VirtualHomeHeuristic(), domain, state)
hval = heuristic(domain, plan_end_state, true_goal_spec)
planner = AStarPlanner(heuristic, max_nodes=2^14, save_search=true, verbose=true)
sol = planner(domain, plan_end_state, true_goal_spec)

# Test literal command enumeration
s_actions, s_agents, s_predicates =
    enumerate_salient_actions(domain, plan_end_state)
commands = enumerate_commands(s_actions, s_agents, s_predicates)
commands = unique!([lift_command(cmd, plan_end_state) for cmd in commands])
command_strs = [repr("text/llm", cmd) for cmd in commands]

# Test pragmatic command enumeration
cmd_state = copy(plan_end_state)
cmd_state[pddl"(nograb actor)"] = true
planner = AStarPlanner(heuristic, max_nodes=2^16, h_mult=2.0,
                       save_search=true, verbose=true)
cmd_sol = planner(domain, cmd_state, true_goal_spec)
s_actions, s_agents, s_predicates =
    extract_salient_actions(domain, plan_end_state, cmd_sol.plan)
commands = enumerate_commands(s_actions, s_agents, s_predicates)
commands = unique!([lift_command(cmd, plan_end_state) for cmd in commands])
command_strs = [repr("text/llm", cmd) for cmd in commands]

## Run literal listener inference ##

# Infer distribution over commands
commands, command_probs, command_scores =
    literal_command_inference(domain, plan_end_state,
                              utterances, verbose=true)
top_command = commands[1]


# Print top 5 commands and their probabilities
println("Top 5 most probable commands:")
for idx in 1:5
    command_str = repr("text/plain", commands[idx])
    @printf("%.3f: %s\n", command_probs[idx], command_str)
end

# Compute assistance options and plans for top command
top_assist_results = literal_assistance(
    top_command, domain, plan_end_state, true_goal_spec;
    true_assist_options, assist_objs, verbose = true
)

# Compute expected assistance options and plans via systematic sampling
expected_assist_results = literal_assistance(
    commands, command_probs, domain, plan_end_state, true_goal_spec;
    true_assist_options, assist_objs, verbose = true, n_samples = 10
)

## Configure agent and world model ##

# Set options that vary across runs
ACT_TEMPERATURES = 2 .^ collect(-3:0.25:5)
MODALITIES = (:action, :utterance)

# Define goal prior
@gen function goal_prior()
    # Sample goal index
    goal ~ uniform_discrete(1, length(goals))
    # Construct goal specification
    spec = MinStepsGoal(goals[goal])
    return spec
end

# Configure planner
heuristic = VirtualHomeHeuristic()
heuristic = precomputed(heuristic, domain, state)
planner = RTHS(heuristic=heuristic, n_iters=0, reuse_paths=false,
               max_nodes=2^16, fail_fast=true)

# Define communication and action configuration
act_config = HierarchicalBoltzmannActConfig(ACT_TEMPERATURES,
                                            inv_gamma, (1.0, 2.0))
# act_config = HierarchicalBoltzmannActConfig([1.0])
if :utterance in MODALITIES
    act_config = CommunicativeActConfig(
        act_config, # Action configuration
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

# Construct iterator over goals for stratified sampling
goal_addr = :init => :agent => :goal => :goal
init_strata = choiceproduct((goal_addr, 1:length(goals)))

## Run inference on observed actions and utterances ##

# Add do-operator around helper actions
obs_plan = map(plan) do act
    act.args[1] == pddl"(helper)" ? InversePlanning.do_op(act) : act
end

# Convert plan to action choicemaps
observations = act_choicemap_vec(obs_plan)
timesteps = collect(1:length(observations))

# Add utterances to choicemaps
if :utterance in MODALITIES
    # Set `speak` to false for all timesteps
    for (t, obs) in zip(timesteps, observations)
        obs[:timestep => t => :act => :speak] = false
    end
    # Add initial choice map
    init_obs = choicemap((:init => :act => :speak, false))
    pushfirst!(observations, init_obs)
    pushfirst!(timesteps, 0)
    # Constrain `speak` and `utterance` for each timestep where speech occurs
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
    end
end

# For only data logging and printing, use these callbacks
logger_cb = DataLoggerCallback(
    t = (t, pf) -> t::Int,
    goal_probs = pf -> probvec(pf, goal_addr, 1:length(goals))::Vector{Float64},
    lml_est = pf -> log_ml_estimate(pf)::Float64,
)
print_cb = PrintStatsCallback(
    (goal_addr, 1:length(goals)),
    header = ("t\t" * join(shorten_label.(goal_labels), "\t"))
)
callback = CombinedCallback(logger=logger_cb, print=print_cb)

# Configure SIPS particle filter
sips = SIPS(world_config, resample_cond=:none, rejuv_cond=:none)

# Run particle filter to perform online goal inference
n_samples = length(init_strata)
pf_state = sips(
    n_samples,  observations;
    init_args=(init_strata=init_strata,),
    callback=callback
);

# Extract goal probabilities
goal_probs = callback.logger.data[:goal_probs]
goal_probs = reduce(hcat, goal_probs)

## Compute pragmatic assistance options and plans ##

# Assistance by following policy to most likely goal
pragmatic_assist_pmdp_mode_results = pragmatic_assistance_pmdp(
    pf_state, domain, plan_end_state, true_goal_spec;
    true_plan = completion, true_assist_options, assist_objs,
    estim_type = "mode", max_steps=50, verbose = true
)

# Assistance by sampling a goal and following the corresponding policy
pragmatic_assist_pmdp_mean_results = pragmatic_assistance_pmdp(
    pf_state, domain, plan_end_state, true_goal_spec;
    true_plan = completion, true_assist_options, assist_objs,
    estim_type = "mean", max_steps=50, verbose = true
)

# Assistance via expected cost minimization
pragmatic_assist_qmdp_act_results = pragmatic_assistance_qmdp_act(
    pf_state, domain, plan_end_state, true_goal_spec;
    true_plan = completion, true_assist_options, assist_objs,
    max_steps=50, verbose = true
)

# Assistance via expected cost minimization with belief updating
pragmatic_assist_qmdp_act_results = pragmatic_assistance_qmdp_act(
    pf_state, domain, plan_end_state, true_goal_spec;
    true_plan = completion, true_assist_options, assist_objs,
    max_steps=50, verbose = true,
    update_beliefs=true, sips_config = sips
)
