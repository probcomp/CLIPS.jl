using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using InversePlanning
using Printf
# using PDDLViz, GLMakie

using GenParticleFilters: softmax

include("src/utils.jl")
include("src/plan_io.jl")
include("src/heuristics.jl")
include("src/utterances.jl")
include("src/inference.jl")
include("src/assistance.jl")
# include("src/render.jl")
# include("src/callbacks.jl")

PDDL.Arrays.@register()
# GLMakie.activate!(inline=false)

## Load domains, problems and plans ##

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans", "observed")
COMPLETION_DIR = joinpath(@__DIR__, "dataset", "plans", "completed")
LITERAL_UTTERANCES_DIR = joinpath(@__DIR__, "dataset", "utterances", "literal")

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

# Load utterance-annotated plans and completions
PLAN_IDS, PLANS, UTTERANCES, UTTERANCE_TIMES = load_plan_dataset(PLAN_DIR)
PLAN_IDS, COMPLETIONS, _, _ = load_plan_dataset(COMPLETION_DIR)

# Load simplified utterances for literal listener
_, LITERAL_UTTERANCES = load_utterance_dataset(LITERAL_UTTERANCES_DIR)

# Define possible cost profiles
COST_PROFILES = [
    ( # Human pickup is costly
        human = (
            pickup=5.0, unlock=1.0, handover=1.0, 
            up=1.0, down=1.0, left=1.0, right=1.0, wait=0.6
        ),
        robot = (
            pickup=1.0, unlock=1.0, handover=1.0, 
            up=1.0, down=1.0, left=1.0, right=1.0, wait=0.6
        )
    ),
    ( # Human pickup is costly, human actions more costly
        human = (
            pickup=5.0, unlock=2.0, handover=2.0, 
            up=2.0, down=2.0, left=2.0, right=2.0, wait=0.6
        ),
        robot = (
            pickup=1.0, unlock=1.0, handover=1.0, 
            up=1.0, down=1.0, left=1.0, right=1.0, wait=0.6
        )
    ),
    ( # Human pickup is costly, robot unlock is costly
        human = (
            pickup=5.0, unlock=1.0, handover=1.0, 
            up=1.0, down=1.0, left=1.0, right=1.0, wait=0.6
        ),
        robot = (
            pickup=1.0, unlock=5.0, handover=1.0, 
            up=1.0, down=1.0, left=1.0, right=1.0, wait=0.6
        )
    ),
    ( # Human pickup and robot unlock are costly, human actions more costly
        human = (
            pickup=5.0, unlock=2.0, handover=2.0, 
            up=2.0, down=2.0, left=2.0, right=2.0, wait=0.6
        ),
        robot = (
            pickup=1.0, unlock=5.0, handover=1.0, 
            up=1.0, down=1.0, left=1.0, right=1.0, wait=0.6
        )
    ),
]

## Set-up for specific plan and problem ##

# Select plan and problem
plan_id = "1.1.keys"

plan = PLANS[plan_id]
utterances = UTTERANCES[plan_id]
utterance_times = UTTERANCE_TIMES[plan_id]
literal_utterance = LITERAL_UTTERANCES[plan_id]

assist_type = match(r"(\d+\w?).(\d+)\.(\w+)", plan_id).captures[3]
assist_obj_type = assist_type == "keys" ? :key : :door

problem_id = match(r"(\d+\w?).(\d+)\.(\w+)", plan_id).captures[1]
problem = PROBLEMS[problem_id]

# Determine true goal from completion
completion = COMPLETIONS[plan_id]
true_goal_obj, true_goal = extract_goal(completion)

# Construct true goal specification
action_costs = assist_type == "doors" ? COST_PROFILES[1] :  COST_PROFILES[3]
true_goal_spec = MinPerAgentActionCosts(Term[true_goal], action_costs)

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
true_assist_options, option_count =
    extract_assist_options(state, completion, assist_type)

## Run literal listener inference ##

# Infer distribution over commands
commands, command_probs, command_scores =
    literal_command_inference(domain, plan_end_state,
                              literal_utterance, verbose=true)
top_command = commands[1]

# Print top 5 commands and their probabilities
println("Top 5 most probable commands:")
for idx in 1:5
    command_str = repr("text/plain", commands[idx])
    @printf("%.3f: %s\n", command_probs[idx], command_str)
end

# Compute naive assistance options and plans for top command
top_naive_assist_results = literal_assistance_naive(
    top_command, domain, plan_end_state, true_goal_spec, assist_obj_type;
    true_assist_options, verbose = true
)

# Compute expected assistance options and plans via systematic sampling
expected_naive_assist_results = literal_assistance_naive(
    commands, command_probs,
    domain, plan_end_state, true_goal_spec, assist_obj_type;
    true_assist_options, verbose = true, n_samples = 10
)

# Compute efficient assistance options and plans for top command
top_efficient_assist_results = literal_assistance_efficient(
    top_command, domain, plan_end_state, true_goal_spec, assist_obj_type;
    true_assist_options, verbose = true
)

# Compute expected assistance options and plans via systematic sampling
expected_efficient_assist_results = literal_assistance_efficient(
    commands, command_probs,
    domain, plan_end_state, true_goal_spec, assist_obj_type;
    true_assist_options, verbose = true, n_samples = 10
)

## Configure agent and world model ##

# Set options that vary across runs
ACT_TEMPERATURES = 2 .^ collect(-3:0.25:5)
MODALITIES = (:action, :utterance)

# Define possible goals
goals = @pddl("(has human gem1)", "(has human gem2)",
              "(has human gem3)", "(has human gem4)")
goal_names = ["red", "yellow", "blue", "green"]

# Select cost profiles based on assistance type
if assist_type == "doors"
    cost_profiles = COST_PROFILES[1:2]
elseif assist_type == "keys"
    cost_profiles = COST_PROFILES[3:4]
end

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
planner = RTHS(heuristic=heuristic, n_iters=1, max_nodes=2^18, fail_fast=true)

# Define communication and action configuration
act_config = HierarchicalBoltzmannActConfig(ACT_TEMPERATURES,
                                            inv_gamma, (0.5, 1.0))
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

# Construct iterator over goals and cost profiles for stratified sampling
goal_addr = :init => :agent => :goal => :goal
cost_addr = :init => :agent => :goal => :cost_idx
init_strata = choiceproduct(
    (goal_addr, 1:length(goals)),
    (cost_addr, 1:length(cost_profiles)),
)

## Run inference on observed actions and utterances ##

# Add do-operator around robot actions
obs_plan = map(plan) do act
    act.args[1] == pddl"(robot)" ? InversePlanning.do_op(act) : act
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

# Construct callback for logging data and visualizing inference
# renderer = get(renderer_dict, assist_type, RENDERER)
# callback = DKGCombinedCallback(
#     renderer, domain;
#     goal_addr = goal_addr,
#     goal_names = goal_names,
#     goal_colors = gem_colors,
#     obs_trajectory = PDDL.simulate(domain, state, plan),
#     print_goal_probs = true,
#     plot_goal_bars = false,
#     plot_goal_lines = false,
#     render = true,
#     inference_overlay = true,
#     record = false
# )

# For only data logging and printing, use these callbacks
logger_cb = DataLoggerCallback(
    t = (t, pf) -> t::Int,
    goal_probs = pf -> probvec(pf, goal_addr, 1:length(goals))::Vector{Float64},
    cost_probs = pf -> probvec(pf, cost_addr, 1:length(cost_profiles))::Vector{Float64},
    lml_est = pf -> log_ml_estimate(pf)::Float64,
)
print_cb = PrintStatsCallback(
    (goal_addr, 1:length(goals)),
    (cost_addr, 1:length(cost_profiles));
    header=("t\t" * join(goal_names, "\t") * "\t" *
            join(["C$C" for C in 1:length(cost_profiles)], "\t") * "\t")
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

# Extract cost probabilities
cost_probs = callback.logger.data[:cost_probs]
cost_probs = reduce(hcat, cost_probs)

## Compute pragmatic assistance options and plans ##

# Assistance by following policy to most likely goal
pragmatic_assist_pmdp_mode_results = pragmatic_assistance_pmdp(
    pf_state, domain, plan_end_state, true_goal_spec, assist_obj_type;
    true_plan = completion, true_assist_options,
    estim_type = "mode", max_steps=100, verbose = true
)

# Assistance by sampling a goal and following the corresponding policy
pragmatic_assist_pmdp_mean_results = pragmatic_assistance_pmdp(
    pf_state, domain, plan_end_state, true_goal_spec, assist_obj_type;
    true_plan = completion, true_assist_options,
    estim_type = "mean", max_steps=100, verbose = true
)

# Assistance via expected cost minimization
pragmatic_assist_qmdp_act_results = pragmatic_assistance_qmdp_act(
    pf_state, domain, plan_end_state, true_goal_spec, assist_obj_type;
    true_plan = completion, true_assist_options,
    max_steps=100, verbose = true
)

# Assistance via expected cost minimization with belief updating
pragmatic_assist_qmdp_act_results = pragmatic_assistance_qmdp_act(
    pf_state, domain, plan_end_state, true_goal_spec, assist_obj_type;
    true_plan = completion, true_assist_options,
    max_steps=100, verbose = true,
    update_beliefs=true, sips_config = sips
)

# Assistance via expected cost minimization over plans
pragmatic_assist_qmdp_plan_mode_results = pragmatic_assistance_qmdp_plan(
    pf_state, domain, plan_end_state, true_goal_spec, assist_obj_type;
    true_assist_options, max_steps=100,
    estim_type="mode", verbose = true
)

# Assistance via soft expected cost minimization over plans
pragmatic_assist_qmdp_plan_mean_results = pragmatic_assistance_qmdp_plan(
    pf_state, domain, plan_end_state, true_goal_spec, assist_obj_type;
    true_assist_options, max_steps=100, rerank_temperature = 0.5,
    estim_type="mean", verbose = true
)
