using PDDL, SymbolicPlanners

"""
    ActionCommand

A sequence of one or more actions to be executed, with optional predicate
modifiers for action arguments.
"""
struct ActionCommand
    "Actions to be executed."
    actions::Vector{Term}
    "Predicate modifiers for action arguments."
    predicates::Vector{Term}
    "Free variables in lifted actions."
    vars::Vector{Var}
    "Types of free variables."
    vtypes::Vector{Symbol}
end

ActionCommand(actions, predicates) =
    ActionCommand(actions, predicates, Var[], Symbol[])

Base.hash(cmd::ActionCommand, h::UInt) =
    hash(cmd.vtypes, hash(cmd.vars, hash(cmd.predicates, hash(cmd.actions, h))))

Base.:(==)(cmd1::ActionCommand, cmd2::ActionCommand) =
    cmd1.actions == cmd2.actions && cmd1.predicates == cmd2.predicates &&
    cmd1.vars == cmd2.vars && cmd1.vtypes == cmd2.vtypes

Base.issetequal(cmd1::ActionCommand, cmd2::ActionCommand) =
    issetequal(cmd1.actions, cmd2.actions) &&
    issetequal(cmd1.predicates, cmd2.predicates) &&
    length(cmd1.vars) == length(cmd2.vars) &&
    Dict(zip(cmd1.vars, cmd1.vtypes)) == Dict(zip(cmd2.vars, cmd2.vtypes))

Base.issubset(cmd1::ActionCommand, cmd2::ActionCommand) =
    issubset(cmd1.actions, cmd2.actions) &&
    issubset(cmd1.predicates, cmd2.predicates) &&
    issubset(Dict(zip(cmd1.vars, cmd1.vtypes)), Dict(zip(cmd2.vars, cmd2.vtypes)))

function Base.merge(cmd1::ActionCommand, cmd2::ActionCommand)
    actions = unique!(vcat(cmd1.actions, cmd2.actions))
    predicates = unique!(vcat(cmd1.predicates, cmd2.predicates))
    vdict1 = Dict(zip(cmd1.vars, cmd1.vtypes))
    vdict2 = Dict(zip(cmd2.vars, cmd2.vtypes))
    vdict = merge(vdict1, vdict2)
    vars = collect(keys(vdict))
    vtypes = collect(values(vdict))
    return ActionCommand(actions, predicates, vars, vtypes)
end

function Base.show(io::IO, ::MIME"text/plain", command::ActionCommand)
    action_str = join(write_pddl.(command.actions), " ")
    print(io, action_str)
    if !isempty(command.predicates)
        print(io, " where ")
        pred_str = join(write_pddl.(command.predicates), " ")
        print(io, pred_str)
    end
end

function Base.show(io::IO, ::MIME"text/llm", command::ActionCommand)
    command = type_to_itemtype(drop_unpredicated_vars(command))
    action_str = join(write_pddl.(command.actions), " ")
    action_str = filter(!=('?'), action_str)
    print(io, action_str)
    if !isempty(command.predicates)
        print(io, " where ")
        pred_str = join(write_pddl.(command.predicates), " ")
        pred_str = filter(!=('?'), pred_str)
        print(io, pred_str)
    end
end

"Replace arguments in an action command with variables."
function lift_command(command::ActionCommand, state::State;
                      ignore = [pddl"(me)", pddl"(you)"],
                      type_name_fn = (s, o) -> string(get_item_type(s, o)))
    args_to_vars = Dict{Const, Var}()
    name_count = Dict{String, Int}()
    vars = Var[]
    vtypes = Symbol[]
    actions = map(command.actions) do act
        args = map(act.args) do arg
            arg in ignore && return arg
            arg isa Const || return arg
            var = get!(args_to_vars, arg) do
                type = PDDL.get_objtype(state, arg)
                name = type_name_fn(state, arg)
                count = get(name_count, name, 0) + 1
                name_count[name] = count
                name = Symbol(uppercasefirst(name), count)
                v = Var(name)
                push!(vars, v)
                push!(vtypes, type)
                return v
            end
            return var
        end
        return Compound(act.name, args)
    end
    predicates = map(command.predicates) do pred
        pred isa Compound || return pred
        args = map(pred.args) do arg
            arg isa Const || return arg
            return get(args_to_vars, arg, arg)
        end
        return Compound(pred.name, args)
    end
    return ActionCommand(actions, predicates, vars, vtypes)
end

"Grounds a lifted action command into a set of ground action commands."
function ground_command(command::ActionCommand, domain::Domain, state::State)
    vars = command.vars
    types = command.vtypes
    # Return original command if it has no variables
    length(vars) == 0 && return [command]
    # Find all possible groundings
    typeconds = Term[Compound(ty, Term[var]) for (ty, var) in zip(types, vars)]
    neqconds = Term[Compound(:not, [Compound(:(==), Term[vars[i], vars[j]])])
                    for i in eachindex(vars) for j in 1:i-1]
    conds = [command.predicates; neqconds; typeconds]
    substs = satisfiers(domain, state, conds)
    g_commands = ActionCommand[]
    for s in substs
        actions = map(act -> PDDL.substitute(act, s), command.actions)
        predicates = map(pred -> PDDL.substitute(pred, s), command.predicates)
        push!(g_commands, ActionCommand(actions, predicates))
    end
    # Remove duplicate groundings according to set equality
    unique_g_commands = ActionCommand[]
    for g_cmd in g_commands
        if !any(issetequal(g_cmd, ug_cmd) for ug_cmd in unique_g_commands)
            push!(unique_g_commands, g_cmd)
        end
    end
    return unique_g_commands
end

"Converts object types in a command to item types."
function type_to_itemtype(command::ActionCommand)
    vars = Var[]
    vtypes = Symbol[]
    predicates = Term[]
    for pred in command.predicates
        if pred.name == :itemtype
            push!(vars, pred.args[1])
            push!(vtypes, pred.args[2].name)
        else
            push!(predicates, pred)
        end
    end
    for (var, ty) in zip(command.vars, command.vtypes)
        var in vars && continue
        push!(vars, var)
        push!(vtypes, ty)
    end
    return ActionCommand(command.actions, predicates, vars, vtypes)
end

"Drop variables from an action command that are not predicated."
function drop_unpredicated_vars(command::ActionCommand)
    predicated = Var[]
    for pred in command.predicates
        for arg in pred.args
            arg isa Var || continue
            push!(predicated, arg)
        end
    end
    unpredicated = setdiff(command.vars, predicated)
    actions = map(command.actions) do act
        args = filter(!in(unpredicated), act.args)
        return Compound(act.name, args)
    end
    vars = Var[]
    types = Symbol[]
    for (var, ty) in zip(command.vars, command.vtypes)
        var in unpredicated && continue
        push!(types, ty)
    end
    return ActionCommand(actions, command.predicates, vars, types)
end

"Statically check if a ground action command is possible in a domain and state."
function is_command_possible(
    command::ActionCommand, domain::Domain, state::State;
    speaker = pddl"(actor)", listener = pddl"(helper)",
    statics = PDDL.infer_static_fluents(domain)
)
    possible = true
    command = pronouns_to_names(command; speaker, listener)
    for act in command.actions
        # Substitute and simplify precondition
        precond = PDDL.get_precond(domain, act)
        # Append predicates
        preconds = append!(PDDL.flatten_conjs(precond), command.predicates)
        precond = Compound(:and, preconds)
        # Simplify static terms
        precond = PDDL.dequantify(precond, domain, state)
        precond = PDDL.simplify_statics(precond, domain, state, statics)
        possible = precond.name != false
    end
    return possible
end

"Convert an action command to one or more goal formulas."
function command_to_goals(
    command::ActionCommand;
    speaker = pddl"(actor)",
    listener = pddl"(helper)",
    act_goal_map = Dict(
        :grab => act -> begin
            item = act.args[2]
            if item isa Var
                item = Const(Symbol(lowercase(string(item.name))))
            end
            return Compound(:placed, Term[item, pddl"(table1)"])
        end
    )
)
    vars = command.vars
    types = command.vtypes
    # Convert each action to goal
    goals = map(command.actions) do action
        action = pronouns_to_names(action; speaker, listener)
        goal = act_goal_map[action.name](action)
        return PDDL.flatten_conjs(goal)        
    end
    goals = reduce(vcat, goals)
    # Convert to existential quantifier if variables are present
    if !all(PDDL.is_ground.(goals)) && !isempty(vars)
        # Find predicate constraints which include some variables
        constraints = filter(pred -> any(arg in vars for arg in pred.args),
                             command.predicates)
        # Construct type constraints
        typeconds = Term[Compound(ty, Term[v]) for (ty, v) in zip(types, vars)]
        typecond = length(typeconds) > 1 ?
            Compound(:and, typeconds) : typeconds[1]
        neqconds = Term[Compound(:not, [Compound(:(==), Term[vars[i], vars[j]])])
                        for i in eachindex(vars) for j in 1:i-1]
        # Construct existential quantifier
        body = Compound(:and, append!(goals, neqconds, constraints))
        goals = [Compound(:exists, Term[typecond, body])]
    end
    return goals
end

"Replace speaker and listener names with pronouns."
function names_to_pronouns(
    command::ActionCommand;
    speaker = pddl"(actor)", listener = pddl"(helper)"
)
    actions = map(command.actions) do act
        names_to_pronouns(act; speaker, listener)
    end
    predicates = map(command.predicates) do pred
        names_to_pronouns(pred; speaker, listener)
    end
    return ActionCommand(actions, predicates, command.vars, command.vtypes)
end

function names_to_pronouns(
    term::Compound;
    speaker = pddl"(actor)", listener = pddl"(helper)"
)
    new_args = map(term.args) do arg
        if arg == speaker
            pddl"(me)"
        elseif arg == listener
            pddl"(you)"
        else
            names_to_pronouns(arg; speaker, listener)
        end
    end
    return Compound(term.name, new_args)
end
names_to_pronouns(term::Var; kwargs...) = term
names_to_pronouns(term::Const; kwargs...) = term

"Replace pronouns with speaker and listener names."
function pronouns_to_names(
    command::ActionCommand;
    speaker = pddl"(actor)", listener = pddl"(helper)"
)
    actions = map(command.actions) do act
        pronouns_to_names(act; speaker, listener)
    end
    predicates = map(command.predicates) do pred
        pronouns_to_names(pred; speaker, listener)
    end
    return ActionCommand(actions, predicates, command.vars, command.vtypes)
end

function pronouns_to_names(
    term::Compound;
    speaker = pddl"(actor)", listener = pddl"(helper)"
)
    new_args = map(term.args) do arg
        if arg == pddl"(me)"
            speaker
        elseif arg == pddl"(you)"
            listener
        else
            pronouns_to_names(arg; speaker, listener)
        end
    end
    return Compound(term.name, new_args)
end
pronouns_to_names(term::Var; kwargs...) = term
pronouns_to_names(term::Const; kwargs...) = term


"Extract focal objects from an action command or plan."
function extract_focal_objects(
    command::ActionCommand;
    obj_arg_idxs = Dict(:pickup => 2, :handover => 3, :unlock => 3)
)
    objects = Term[]
    for act in command.actions
        act.name in keys(obj_arg_idxs) || continue
        idx = obj_arg_idxs[act.name]
        obj = act.args[idx]
        push!(objects, obj)
    end
    return unique!(objects)
end

function extract_focal_objects(
    plan::AbstractVector{<:Term};
    obj_arg_idxs = Dict(:grab => 2),
    listener = pddl"(helper)"
)
    focal_objs = Term[]
    for act in plan
        act.args[1] == listener || continue
        act.name in keys(obj_arg_idxs) || continue
        idx = obj_arg_idxs[act.name]
        obj = act.args[idx]
        push!(focal_objs, obj)
    end
    return unique!(focal_objs)
end

"Extract focal objects from a plan that matches a lifted action command."
function extract_focal_objects_from_plan(
    command::ActionCommand, plan::AbstractVector{<:Term};
    obj_arg_idxs = Dict(:grab => 2),
    speaker = pddl"(actor)", listener = pddl"(helper)"
)
    focal_vars = extract_focal_objects(command; obj_arg_idxs)
    focal_objs = Term[]
    command = pronouns_to_names(command; speaker, listener)
    for cmd_act in command.actions
        for act in plan
            unifiers = PDDL.unify(cmd_act, act)
            isnothing(unifiers) && continue
            for var in focal_vars
                if haskey(unifiers, var)
                    push!(focal_objs, unifiers[var])
                end
            end
        end
    end
    return unique!(focal_objs)
end
