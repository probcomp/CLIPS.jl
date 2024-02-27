using PDDL
using SymbolicPlanners
using PDDLViz
using JSON3

PDDL.Arrays.@register()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/render.jl")

"Generates stimulus plan completion."
function generate_stim_completion(
    path::Union{AbstractString, Nothing},
    domain::Domain,
    problem::Problem,
    plan::AbstractVector{<:Term};
    kwargs...
)
    state = initstate(domain, problem)
    goal = PDDL.get_goal(problem)
    return generate_stim_completion(path, domain, state, plan, goal; kwargs...)
end

function generate_stim_completion(
    path::Union{AbstractString, Nothing},
    domain::Domain,
    state::State,
    plan::AbstractVector{<:Term},
    goal::Union{Term, AbstractVector{<:Term}};
    action_costs = (
        human=(
            pickup=5.0, unlock=1.0, handover=1.0, 
            up=1.0, down=1.0, left=1.0, right=1.0, wait=0.6
        ),
        robot = (
            pickup=1.0, unlock=5.0, handover=1.0, 
            up=1.0, down=1.0, left=1.0, right=1.0, wait=0.6
        )
    )
)
    # Simulate state after observed plan has been executed
    state = PDDL.simulate(PDDL.EndStateSimulator(), domain, state, plan)
    # Construct planner and goal specification
    planner = AStarPlanner(DoorsKeysMSTHeuristic())
    spec = MinPerAgentActionCosts(PDDL.flatten_conjs(goal), action_costs)
    # Solve for plan completion
    sol = planner(domain, state, spec)
    completion = collect(sol)
    # Save plan completion to file
    if !isnothing(path)
        save_plan(path, completion)
    end
    return completion
end

"Generates stimulus animation for an utterance-annotated plan."
function generate_stim_anim(
    path::Union{AbstractString, Nothing},
    domain::Domain,
    state::State,
    plan::AbstractVector{<:Term},
    utterances::AbstractVector{<:AbstractString} = String[],
    utterance_times::AbstractVector{Int} = Int[];
    assist_type = "none",
    renderer = renderer_dict[assist_type],
    caption_prefix = "Human: ",
    caption_quotes = "\"",
    caption_line_length = 40,
    caption_lines = 1,
    caption_color = :black,
    caption_dur = 4,
    caption_size = 28,
    framerate = 3,
    trail_length = 15,
    format = "gif",
    loop = -1,
)
    # Determine number of lines
    for u in utterances
        len = length(u) + length(caption_prefix) + 2*length(caption_quotes)
        n_lines = (len-1) ÷ caption_line_length + 1
        caption_lines = max(caption_lines, n_lines)
    end
    # Preprocess utterances into multiple lines
    delims = ['?', '!', '.', ',', ';', ' ']
    utterances = map(utterances) do u
        u = caption_prefix * caption_quotes * u * caption_quotes
        lines = String[]
        l_start = 1
        count = 0
        while l_start <= length(u) && count < caption_lines
            count += 1
            l_stop_max = min(l_start + caption_line_length - 1, length(u))
            if l_stop_max >= length(u)
                push!(lines, u[l_start:end])
                break
            end
            l_stop = nothing
            for d in delims
                l_stop = findlast(d, u[l_start:l_stop_max])
                !isnothing(l_stop) && break
            end
            l_stop = isnothing(l_stop) ? l_stop_max : l_stop + l_start - 1
            push!(lines, u[l_start:l_stop])
            l_start = l_stop + 1
        end
        if length(lines) < caption_lines
            append!(lines, fill("", caption_lines - length(lines)))
        end
        u = join(lines, "\n")
        return u
    end
    # Construct caption dictionary
    blank_str = join(fill("...", caption_lines), "\n")
    if isempty(utterances)
        captions = Dict(1 => blank_str)
        caption_color = :white
    else
        captions = Dict(1 => blank_str)
        for (t, u) in zip(utterance_times, utterances)
            captions[t+1] = u
            if !any(t+1 <= k <= t+caption_dur for k in keys(captions))
                captions[t+1+caption_dur] = blank_str
            end
        end
    end
    # Animate plan
    anim = anim_plan(renderer, domain, state, plan;
                     captions, caption_color, caption_size,
                     trail_length, framerate, format, loop)
    # Save animation
    if !isnothing(path)
        save(path, anim)
    end
    return anim
end

function generate_stim_anim(
    path::Union{AbstractString, Nothing},
    domain::Domain,
    problem::Problem,
    plan::AbstractVector{<:Term},
    utterances::AbstractVector{<:AbstractString} = String[],
    utterance_times::AbstractVector{Int} = Int[];
    kwargs...
)
    state = initstate(domain, problem)
    return generate_stim_anim(path, domain, state, plan,
                              utterances, utterance_times; kwargs...)
end

"Generates stimulus animation set for an utterance-annotated plan."
function generate_stim_anim_set(
    path::Union{AbstractString, Nothing},
    domain::Domain,
    problem::Problem,
    plan::AbstractVector{<:Term},
    completion::AbstractVector{<:Term},
    utterances::AbstractVector{<:AbstractString} = String[],
    utterance_times::AbstractVector{Int} = Int[];
    assist_type = "none",
    framerate = 3,
    kwargs...
)
    # Determine number of lines
    caption_lines = get(kwargs, :caption_lines, 1)
    caption_line_length = get(kwargs, :caption_line_length, 40)
    caption_prefix = get(kwargs, :caption_prefix, "Human: ")
    caption_quotes = get(kwargs, :caption_quotes, "\"")
    for u in utterances
        len = length(u) + length(caption_prefix) + 2*length(caption_quotes)
        n_lines = (len-1) ÷ caption_line_length + 1
        caption_lines = max(caption_lines, n_lines)
    end
    name, ext = splitext(path)
    state = initstate(domain, problem)
    # Generate initial frame
    init_path = name * "_0_init" * ext
    println("Generating $init_path...")
    init_anim = generate_stim_anim(
        init_path, domain, state, Term[], utterances, utterance_times;
        assist_type, caption_lines, kwargs...
    )
    # Generate plan animation
    plan_path = name * "_1_observed" * ext
    println("Generating $plan_path...")
    plan_anim = generate_stim_anim(
        plan_path, domain, state, plan, utterances, utterance_times;
        assist_type, framerate, caption_lines, kwargs...
    )
    # Generate completion animation
    state = PDDL.simulate(PDDL.EndStateSimulator(), domain, state, plan)
    completion_path = name * "_2_completed" * ext
    println("Generating $completion_path...")
    completion_anim = generate_stim_anim(
        completion_path, domain, state, completion;
        assist_type, framerate=framerate+1, caption_lines, kwargs...
    )
    return [init_anim, plan_anim, completion_anim]
end

"Generate stimuli JSON dictionary."
function generate_stim_json(
    name::String,
    domain::Domain,
    problem::Problem,
    plan::AbstractVector{<:Term},
    completion::AbstractVector{<:Term},
    utterances::AbstractVector{<:AbstractString} = String[];
    assist_type = match(r".*\.(\w+)", name).captures[1]
)
    state = initstate(domain, problem)
    if assist_type == "keys"
        option_count = length(PDDL.get_objects(state, :key))
    else
        option_count = length(PDDL.get_objects(state, :door))
    end
    goal_obj, _ = extract_goal(completion)
    goal_idx = parse(Int, string(goal_obj)[end]) - 1
    options, option_count =
        extract_assist_options(state, completion, assist_type)
    option_idxs = [parse(Int, string(o)[end]) for o in options]
    json = (
        name = name,
        images = [
            "$(name)_0_init.gif",
            "$(name)_1_observed.gif",
            "$(name)_2_completed.gif",
        ],
        type = assist_type,
        utterance = isempty(utterances) ? "" : utterances[1],
        timesteps = length(plan),
        option_count = option_count,
        goal = [goal_idx],
        best_option = [option_idxs]
    )
    return json
end

"Read stimulus inputs from plan and problem datasets."
function read_stim_inputs(
    name::String, problems, plans, completions, utterances, utterance_times
)
    m = match(r"^(\w*\d+\w?).(\d+)\.(\w+)", name)
    problem_name = m.captures[1]
    assist_type = m.captures[3]
    problem = problems[problem_name]
    plan = plans[name]
    completion = completions[name]
    utts = utterances[name]
    utt_times = utterance_times[name]
    return problem, plan, completion, utts, utt_times, assist_type
end

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans", "observed")
COMPLETION_DIR = joinpath(@__DIR__, "dataset", "plans", "completed")
STIMULI_DIR = joinpath(@__DIR__, "dataset", "stimuli")

# Load domain
domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

# Load problems
problems = Dict{String, Problem}()
for path in readdir(PROBLEM_DIR)
    name, ext = splitext(path)
    ext == ".pddl" || continue
    problems[name] = load_problem(joinpath(PROBLEM_DIR, path))
end

# Load utterance-annotated plans and completions
pnames, plans, utterances, utterance_times = load_plan_dataset(PLAN_DIR)
pnames_c, completions, _, _ = load_plan_dataset(COMPLETION_DIR)

# Generate stimuli completions if they don't exist
for name in pnames
    name in pnames_c && continue
    # Load problem
    m = match(r"^(\w*\d+\w?).(\d+)\.(\w+)", name)
    problem_name = m.captures[1]
    problem = problems[problem_name]
    # Compile domain for problem
    state = initstate(domain, problem)
    c_domain, _ = PDDL.compiled(domain, state)
    # Generate and save completion
    println("Generating completion for $name...")
    plan = plans[name]
    c_path = joinpath(COMPLETION_DIR, name * ".pddl")
    completion = generate_stim_completion(c_path, c_domain, problem, plan)
end

# Reload completions
pnames_c, completions, _, _ = load_plan_dataset(COMPLETION_DIR)

# Generate stimuli animations
for name in pnames
    println("Generating animations for $name...")
    problem, plan, completion, utts, utt_times, assist_type =
        read_stim_inputs(name, problems, plans, completions,
                         utterances, utterance_times)

    path = joinpath(STIMULI_DIR, name * ".gif")
    generate_stim_anim_set(path, domain, problem, plan, completion,
                           utts, utt_times; assist_type)
    GC.gc()
end

# Generate stimuli metadata
all_metadata = []
for name in pnames
    println("Generating metadata for $name...")
    problem, plan, completion, utts, utt_times, assist_type =
        read_stim_inputs(name, problems, plans, completions,
                         utterances, utterance_times)
    json = generate_stim_json(name, domain, problem, plan, completion, utts;
                              assist_type)
    push!(all_metadata, json)
end
metadata_path = joinpath(STIMULI_DIR, "stimuli.json")
open(metadata_path, "w") do io
    JSON3.pretty(io, all_metadata)
end

# Load demonstration plans and completions
pnames, plans, utterances, utterance_times =
    load_plan_dataset(PLAN_DIR, r"demo(\d+)\.(\d+)\.\w+")
pnames_c, completions, _, _ =
    load_plan_dataset(COMPLETION_DIR, r"demo(\d+)\.(\d+)\.\w+")

# Generate demonstration completions if they don't exist
for name in pnames
    name in pnames_c && continue
    # Load problem
    m = match(r"^(\w*\d+\w?).(\d+)\.(\w+)", name)
    problem_name = m.captures[1]
    problem = problems[problem_name]
    # Compile domain for problem
    state = initstate(domain, problem)
    c_domain, _ = PDDL.compiled(domain, state)
    # Generate and save completion
    println("Generating completion for $name...")
    plan = plans[name]
    c_path = joinpath(COMPLETION_DIR, name * ".pddl")
    completion = generate_stim_completion(c_path, c_domain, problem, plan)
end

# Reload completions
pnames_c, completions, _, _ =
    load_plan_dataset(COMPLETION_DIR, r"demo(\d+)\.(\d+)\.\w+")

# Generate demonstration animations
for name in pnames
    println("Generating animations for $name...")
    problem, plan, completion, utts, utt_times, assist_type =
        read_stim_inputs(name, problems, plans, completions,
                         utterances, utterance_times)

    path = joinpath(STIMULI_DIR, name * ".gif")
    generate_stim_anim_set(path, domain, problem, plan, completion,
                           utts, utt_times; assist_type)
    GC.gc()
end
