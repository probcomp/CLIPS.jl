using PDDL, PDDLViz
using SymbolicPlanners
using InversePlanning

using PDDLViz: RGBA, to_color, set_alpha

"""
    DKGCombinedCallback(renderer, domain; kwargs...)

Convenience constructor for a combined particle filter callback that 
logs data and visualizes inference for the doors, keys and gems domain.

# Keyword Arguments

- `goal_addr`: Trace address of goal variable.
- `goal_names`: Names of goals.
- `goal_colors`: Colors of goals.
- `obs_trajectory = nothing`: Ground truth / observed trajectory.
- `print_goal_probs = true`: Whether to print goal probabilities.
- `render = true`: Whether to render the gridworld.
- `inference_overlay = true`: Whether to render inference overlay.
- `plot_goal_bars = false`: Whether to plot goal probabilities as a bar chart.
- `plot_goal_lines = false`: Whether to plot goal probabilities over time.
- `record = false`: Whether to record the figure.
- `sleep = 0.2`: Time to sleep between frames.
- `framerate = 5`: Framerate of recorded video.
- `format = "mp4"`: Format of recorded video.
"""    
function DKGCombinedCallback(
    renderer::GridworldRenderer, domain::Domain;
    goal_addr = :init => :agent => :goal => :goal,
    goal_names = ["(has gem1)", "(has gem2)", "(has gem3)"],
    goal_colors = PDDLViz.colorschemes[:vibrant][1:length(goal_names)],
    cost_addr = :init => :agent => :goal => :cost_idx,
    n_costs = 4,
    obs_trajectory = nothing,
    print_goal_probs::Bool = true,
    render::Bool = true,
    inference_overlay = true,
    plot_goal_bars::Bool = false,
    plot_goal_lines::Bool = false,
    record::Bool = false,
    sleep::Real = 0.2,
    framerate = 5,
    format = "mp4"
)
    callbacks = OrderedDict{Symbol, SIPSCallback}()
    n_goals = length(goal_names)
    # Construct data logger callback
    callbacks[:logger] = DataLoggerCallback(
        t = (t, pf) -> t::Int,
        goal_probs = pf -> probvec(pf, goal_addr, 1:n_goals)::Vector{Float64},
        cost_probs = pf -> probvec(pf, cost_addr, 1:n_costs)::Vector{Float64},
        lml_est = pf -> log_ml_estimate(pf)::Float64,
    )
    # Construct print callback
    if print_goal_probs
        callbacks[:print] = PrintStatsCallback(
            (goal_addr, 1:n_goals);
            header="t\t" * join(goal_names, "\t") * "\n"
        )
    end
    # Construct render callback
    if render
        figure = Figure(resolution=(600, 700))
        if inference_overlay
            function trace_color_fn(tr)
                goal_idx = tr[goal_addr]
                return goal_colors[goal_idx]
            end
            overlay = DKGInferenceOverlay(trace_color_fn=trace_color_fn)
        end
        callbacks[:render] = RenderCallback(
            renderer, figure[1, 1], domain;
            trajectory=obs_trajectory, trail_length=10,
            overlay = inference_overlay ? overlay : nothing
        )
    end
    # Construct plotting callbacks
    if plot_goal_bars || plot_goal_lines
        if render
            resize!(figure, (1200, 700))
        else
            figure = Figure(resolution=(600, 700))
        end
        side_layout = GridLayout(figure[1, 2])
    end
    if plot_goal_bars
        callbacks[:goal_bars] = BarPlotCallback(
            side_layout[1, 1],
            pf -> probvec(pf, goal_addr, 1:n_goals)::Vector{Float64};
            color = goal_colors,
            axis = (xlabel="Goal", ylabel = "Probability",
                    limits=(nothing, (0, 1)), 
                    xticks=(1:length(goals), goal_names))
        )
    end
    if plot_goal_lines
        callbacks[:goal_lines] = SeriesPlotCallback(
            side_layout[2, 1],
            callbacks[:logger], 
            :goal_probs, # Look up :goal_probs variable
            ps -> reduce(hcat, ps); # Convert vectors to matrix for plotting
            color = goal_colors, labels=goal_names,
            axis = (xlabel="Time", ylabel = "Probability",
                    limits=((1, nothing), (0, 1)))
        )
    end
    # Construct recording callback
    if record && render
        callbacks[:record] = RecordCallback(figure, framerate=framerate,
                                            format=format)
    end
    # Display figure
    if render || plot_goal_bars || plot_goal_lines
        display(figure)
    end
    # Combine all callback functions
    callback = CombinedCallback(;sleep=sleep, callbacks...)
    return callback
end

"""
    DKGInferenceOverlay(; kwargs...)

Inference overlay renderer for the doors, keys and gems domain.

# Keyword Arguments

- `show_state = false`: Whether to show the current estimated state distribution.
- `show_future_states = true`: Whether to show future predicted states.
- `max_future_steps = 50`: Maximum number of future steps to render.
- `trace_color_fn = tr -> :red`: Function to determine the color of a trace.
"""
@kwdef mutable struct DKGInferenceOverlay
    show_state::Bool = false
    show_future_states::Bool = true
    max_future_steps::Int = 50
    trace_color_fn::Function = tr -> :red
    color_obs::Vector = Observable[]
    state_obs::Vector = Observable[]
    future_obs::Vector = Observable[]
end

function (overlay::DKGInferenceOverlay)(
    canvas::Canvas, renderer::GridworldRenderer, domain::Domain,
    t::Int, obs::ChoiceMap, pf_state::ParticleFilterState
)
    traces = get_traces(pf_state)
    weights = get_norm_weights(pf_state)
    # Render future states (skip t = 0 since plans are not yet available) 
    if overlay.show_future_states && t > 0
        for (i, (tr, w)) in enumerate(zip(traces, weights))
            # Get current belief, goal, and plan
            belief_state = tr[:timestep => t => :agent => :belief]
            goal_state = tr[:timestep => t => :agent => :goal]
            plan_state = tr[:timestep => t => :agent => :plan]
            # Get planner from agent configuration
            plan_config = tr[:init][2].agent_config.plan_config
            planner = plan_config.step_args[2]
            # Rollout planning solution until goal is reached
            state = convert(State, belief_state)
            spec = convert(Specification, goal_state)
            sol = plan_state.sol
            future_actions = rollout_sol(domain, planner, state, sol, spec;
                                         max_steps=overlay.max_future_steps)
            future_states = PDDL.simulate(domain, state, future_actions)
            # Render or update future states
            color = overlay.trace_color_fn(tr)
            future_obs = get(overlay.future_obs, i, nothing)
            color_obs = get(overlay.color_obs, i, nothing)
            if isnothing(future_obs)
                future_obs = Observable(future_states)
                color_obs = Observable(to_color((color, w)))
                push!(overlay.future_obs, future_obs)
                push!(overlay.color_obs, color_obs)
                options = renderer.trajectory_options
                object_colors=fill(color_obs, length(options[:tracked_objects]))
                type_colors=fill(color_obs, length(options[:tracked_types]))
                render_trajectory!(
                    canvas, renderer, domain, future_obs;
                    track_markersize=0.5, agent_color=color_obs,
                    object_colors=object_colors, type_colors=type_colors
                )
            else
                future_obs[] = future_states
                color_obs[] = to_color((color, w))
            end
        end
    end
    # Render current state's agent location
    if overlay.show_state
        for (i, (tr, w)) in enumerate(zip(traces, weights))
            # Get current inferred environment state
            env_state = t == 0 ? tr[:init => :env] : tr[:timestep => t => :env]
            state = convert(State, env_state)
            # Construct or update color observable
            color = overlay.trace_color_fn(tr)
            color_obs = get(overlay.color_obs, i, nothing)
            if isnothing(color_obs)
                color_obs = Observable(to_color((color, w)))
            else
                color_obs[] = to_color((color, w))
            end
            # Render or update state
            state_obs = get(overlay.state_obs, i, nothing)
            if isnothing(state_obs)
                state_obs = Observable(state)
                push!(overlay.state_obs, state_obs)
                _trajectory = @lift [$state_obs]
                render_trajectory!(
                    canvas, renderer, domain, _trajectory;
                    agent_color=color_obs, track_markersize=0.6,
                    track_stopmarker='▣' 
                ) 
            else
                state_obs[] = state
            end
        end
    end
end
