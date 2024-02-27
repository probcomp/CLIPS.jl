using PDDL

"""
    load_plan(path::AbstractString)
    load_plan(io::IO)

Load a comment-annotated PDDL plan from file.
"""
function load_plan(io::IO)
    str = read(io, String)
    return parse_plan(str)
end
load_plan(path::AbstractString) = open(io->load_plan(io), path)

"""
    parse_plan(str::AbstractString)

Parse a comment-annotated PDDL plan from a string.
"""
function parse_plan(str::AbstractString)
    plan = Term[]
    annotations = String[]
    annotation_idxs = Int[]
    for line in split(str, "\n")
        line = strip(line)
        if isempty(line)
            continue
        elseif line[1] == ';'
            push!(annotations, strip(line[2:end]))
            push!(annotation_idxs, length(plan))
        else
            push!(plan, parse_pddl(line))
        end
    end
    return plan, annotations, annotation_idxs
end

"""
    save_plan(path, plan, annotations, annotation_idxs)

Save a comment-annotated PDDL plan to file.
"""
function save_plan(
    path::AbstractString,
    plan::AbstractVector{<:Term},
    annotations::AbstractVector{<:AbstractString} = String[],
    annotation_idxs::AbstractVector{Int} = Int[],
)
    str = write_plan(plan, annotations, annotation_idxs)
    open(path, "w") do io
        write(io, str)
    end
    return path
end

"""
    write_plan(plan, annotations, annotation_idxs)

Write a comment-annotated PDDL plan to a string.
"""
function write_plan(
    plan::AbstractVector{<:Term},
    annotations::AbstractVector{<:AbstractString} = String[],
    annotation_idxs::AbstractVector{Int} = Int[],
)
    str = ""
    if 0 in annotation_idxs
        j = findfirst(==(0), annotation_idxs)
        annotation = annotations[j]  
        str *= "; $annotation\n"
    end
    for (i, term) in enumerate(plan)
        str *= write_pddl(term) * "\n"
        if i in annotation_idxs
            j = findfirst(==(i), annotation_idxs)
            annotation = annotations[j]  
            str *= "; $annotation\n"
        end
    end
    return str
end

"""
    load_plan_dataset(dir::AbstractString, [pattern::Regex])

Load utterance-annotated plan dataset from a directory. The `pattern` 
argument is a regular expression that matches the filenames of each plan.
"""
function load_plan_dataset(
    dir::AbstractString, pattern::Regex=r"^(\d+)\.(\d+)\.([\w\d]+)"
)
    paths = readdir(dir)
    filter!(path -> endswith(path, ".pddl"), paths)
    filter!(path -> match(pattern, splitext(path)[1]) !== nothing, paths)
    names = String[]
    plans = Dict{String, Vector{Term}}()
    goals = Dict{String, String}()
    utterances = Dict{String, Vector{String}}()
    utterance_times = Dict{String, Vector{Int}}()
    for path in paths
        m = match(pattern, path)
        name = m.captures[1] * "." * m.captures[2]
        push!(names, name)
        goals[name] = m.captures[3]
        plan, annotations, annotation_idxs = load_plan(joinpath(dir, path))
        plans[name] = plan
        utterances[name] = annotations
        utterance_times[name] = annotation_idxs
    end
    sort!(names, by = x -> (tryparse.(Int, split(x, "."))))
    return names, plans, goals, utterances, utterance_times
end

"""
    load_utterance_dataset(dir::AbstractString, [pattern::Regex])

Load dataset of utterances from a directory. The `pattern` 
argument is a regular expression that matches the filename for each utterance.
"""
function load_utterance_dataset(
    dir::AbstractString, pattern::Regex=r"^(\d+)\.(\d+)\.([\w\d]+)"
)
    paths = readdir(dir)
    filter!(path -> endswith(path, ".txt"), paths)
    filter!(path -> match(pattern, splitext(path)[1]) !== nothing, paths)
    names = String[]
    utterances = Dict{String, String}()
    for path in paths
        name = splitext(path)[1]
        name = m.captures[1] * "." * m.captures[2]
        push!(names, name)
        utt = readline(joinpath(dir, path))
        utterances[name] = utt
    end
    sort!(names, by = x -> (tryparse.(Int, split(x, "."))))
    return names, utterances
end

"""
    load_goal_dataset(dir::AbstractString, [pattern::Regex])

Load dataset of goal definitions from a directory. The `pattern` 
argument is a regular expression that matches the filename for each utterance.
"""
function load_goal_dataset(
    dir::AbstractString, pattern::Regex=r"^\w+"
)
    paths = readdir(dir)
    filter!(path -> endswith(path, ".pddl"), paths)
    filter!(path -> match(pattern, splitext(path)[1]) !== nothing, paths)
    names = String[]
    goals = Dict{String, Term}()
    for path in paths
        name = splitext(path)[1]
        push!(names, name)
        goals[name] = parse_pddl(read(joinpath(dir, path), String))
    end
    sort!(names)
    return names, goals
end
