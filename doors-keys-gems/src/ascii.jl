# Functions for generating gridworld PDDL problems
using PDDL

"Converts ASCII gridworlds to PDDL problem."
function ascii_to_pddl(
    str::String,
    name="doors-keys-gems-problem";
    key_dict = Dict(
        'r' => pddl"(red)",
        'b' => pddl"(blue)",
        'y' => pddl"(yellow)" ,
        'e' => pddl"(green)",
        'p' => pddl"(pink)"
    ),
    door_dict = Dict(
        'R' => pddl"(red)",
        'B' => pddl"(blue)",
        'Y' => pddl"(yellow)" ,
        'E' => pddl"(green)",
        'P' => pddl"(pink)"
    ),
    gem_colors = [pddl"(red)", pddl"(yellow)", pddl"(blue)", pddl"(green)"],
    forbidden = [(pddl"(robot)", :gem)]
)
    rows = split(str, "\n", keepempty=false)
    width, height = maximum(length.(strip.(rows))), length(rows)
    doors, keys, gems, colors = Const[], Const[], Const[], Const[]
    humans, robots = Const[pddl"(human)"], Const[pddl"(robot)"]
    walls = parse_pddl("(= walls (new-bit-matrix false $height $width))")
    init = Term[walls]
    start_human, start_robot, goal = Term[],Term[], pddl"(true)"
    # Parse wall, item, and agent locations
    for (y, row) in enumerate(rows)
        for (x, char) in enumerate(strip(row))
            if char == '.' # Unoccupied
                continue
            elseif char == 'W' # Wall
                wall = parse_pddl("(= walls (set-index walls true $y $x))")
                push!(init, wall)
            elseif haskey(door_dict, char) # Door
                d = Const(Symbol("door$(length(doors)+1)"))
                c = door_dict[char]
                push!(doors, d)
                if !(c in colors)
                    push!(colors, c)
                end
                append!(init, parse_pddl("(= (xloc $d) $x)", "(= (yloc $d) $y)"))
                push!(init, parse_pddl("(iscolor $d $c)"))
                push!(init, parse_pddl("(locked $d)"))
            elseif haskey(key_dict, char)  # Key
                k = Const(Symbol("key$(length(keys)+1)"))
                c = key_dict[char]
                push!(keys, k)
                append!(init, parse_pddl("(= (xloc $k) $x)", "(= (yloc $k) $y)"))
                push!(init, parse_pddl("(iscolor $k $c)"))
            elseif char == 'g' || char == 'G' # Gem
                g = Const(Symbol("gem$(length(gems)+1)"))
                push!(gems, g)
                append!(init, parse_pddl("(= (xloc $g) $x)", "(= (yloc $g) $y)"))
                goal = parse_pddl("(has human $g)")
            elseif char == 'h' # Human
                start_human = parse_pddl("(= (xloc human) $x)", "(= (yloc human) $y)")
            elseif char == 'm' # Robot
                start_robot = parse_pddl("(= (xloc robot) $x)", "(= (yloc robot) $y)")
            end
        end
    end
    append!(init, start_human)
    append!(init, start_robot)
    # Add gem colors
    for g in gems
        c = gem_colors[parse(Int, string(g.name)[end])]
        push!(init, parse_pddl("(iscolor $g $c)"))
        if !(c in colors)
            push!(colors, c)
        end
    end
    # Add turn-taking predicates
    init_turn = parse_pddl(
        "(active human)",
        "(next-turn human robot)",
        "(next-turn robot human)"
    )
    append!(init, init_turn)
    # Add forbidden objects
    for (agent, type) in forbidden
        if type == :gem
            for g in gems
                push!(init, parse_pddl("(forbidden $agent $g)"))
            end
        elseif type == :key
            for k in keys
                push!(init, parse_pddl("(forbidden $agent $k)"))
            end
        end
    end
    objtypes = merge(
        Dict(d => :door for d in doors),
        Dict(k => :key for k in keys),
        Dict(g => :gem for g in gems),
        Dict(c => :color for c in colors),
        Dict(h => :human for h in humans),
        Dict(m => :robot for m in robots)
    )
    problem = GenericProblem(Symbol(name), Symbol("doors-keys-gems"),
                             [doors; keys; gems; colors; robots; humans],
                             objtypes, init, goal,
                             nothing, nothing)
    return problem
end

function load_ascii_problem(path::AbstractString)
    str = open(f->read(f, String), path)
    return ascii_to_pddl(str)
end

function convert_ascii_problem(path::String)
    str = open(f->read(f, String), path)
    str = ascii_to_pddl(str)
    new_path = splitext(path)[1] * ".pddl"
    write(new_path, write_problem(str))
    return new_path
end
