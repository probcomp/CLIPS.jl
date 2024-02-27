# Test gridworld rendering
using PDDL, SymbolicPlanners
using PDDLViz, GLMakie
using Test

using PDDLViz: RGBA, to_color, set_alpha

# Define colors
vibrant = PDDLViz.colorschemes[:vibrant]
gem_colors = [to_color(:red), vibrant[2], colorant"#56b4e9", colorant"#009e73"]
colordict = Dict(
    :red => vibrant[1],
    :yellow => vibrant[2],
    :blue => colorant"#0072b2",
    :green => :springgreen,
    :purple => vibrant[5],
    :orange => vibrant[6],
    :none => :gray
)

# Construct gridworld renderer
RENDERER = PDDLViz.GridworldRenderer(
    resolution = (600, 700),
    has_agent = false,
    obj_renderers = Dict(
        :agent => (d, s, o) -> o.name == :human ?
            HumanGraphic() : RobotGraphic(),
        :key => (d, s, o) -> KeyGraphic(
            color=colordict[get_obj_color(s, o).name]
        ),
        :door => (d, s, o) -> LockedDoorGraphic(
            visible=s[Compound(:locked, [o])],
            color=colordict[get_obj_color(s, o).name]
        ),
        :gem => (d, s, o) -> GemGraphic(
            color=gem_colors[parse(Int, string(o.name)[end])]
        )
    ),
    obj_type_z_order = [:door, :key, :gem, :agent],
    show_inventory = true,
    inventory_fns = [
        (d, s, o) -> s[Compound(:has, [Const(:human), o])],
        (d, s, o) -> s[Compound(:has, [Const(:robot), o])]
    ],
    inventory_types = [:item, :item],
    inventory_labels = ["Human Inventory", "Robot Inventory"],
    trajectory_options = Dict(
        :tracked_objects => [Const(:human), Const(:robot)],
        :tracked_types => Const[],
        :object_colors => [:black, :slategray]
    )
)

# Construct gridworld renderer with labeled keys
RENDERER_LABELED_KEYS = PDDLViz.GridworldRenderer(
    resolution = (600, 700),
    has_agent = false,
    obj_renderers = Dict(
        :agent => (d, s, o) -> o.name == :human ?
            HumanGraphic() : RobotGraphic(),
        :key => (d, s, o) -> MultiGraphic(
            KeyGraphic(-0.1,-0.1,
                color=colordict[get_obj_color(s, o).name]
            ),
            TextGraphic(
                string(o.name)[end:end], 0.3, 0.2, 0.5,
                color=:black, font=:bold
            )
        ),
        :door => (d, s, o) -> LockedDoorGraphic(
            visible=s[Compound(:locked, [o])],
            color=colordict[get_obj_color(s, o).name]
        ),
        :gem => (d, s, o) -> GemGraphic(
            color=gem_colors[parse(Int, string(o.name)[end])]
        )
    ),
    obj_type_z_order = [:door, :key, :gem, :agent],
    show_inventory = true,
    inventory_fns = [
        (d, s, o) -> s[Compound(:has, [Const(:human), o])],
        (d, s, o) -> s[Compound(:has, [Const(:robot), o])]
    ],
    inventory_types = [:item, :item],
    inventory_labels = ["Human Inventory", "Robot Inventory"],
    trajectory_options = Dict(
        :tracked_objects => [Const(:human), Const(:robot)],
        :tracked_types => Const[],
        :object_colors => [:black, :slategray]
    )
)

# Construct gridworld renderer with labeled doors
RENDERER_LABELED_DOORS = PDDLViz.GridworldRenderer(
    resolution = (600, 700),
    has_agent = false,
    obj_renderers = Dict(
        :agent => (d, s, o) -> o.name == :human ?
            HumanGraphic() : RobotGraphic(),
        :key => (d, s, o) -> KeyGraphic(
            color=colordict[get_obj_color(s, o).name]
        ),
        :door => (d, s, o) -> MultiGraphic(
            LockedDoorGraphic(
                visible=s[Compound(:locked, [o])],
                color=colordict[get_obj_color(s, o).name]
            ),
            TextGraphic(
                string(o.name)[end:end], 0.3, 0.2, 0.5,
                color=:white, font=:bold
            )
        ),
        :gem => (d, s, o) -> GemGraphic(
            color=gem_colors[parse(Int, string(o.name)[end])]
        )
    ),
    obj_type_z_order = [:door, :key, :gem, :agent],
    show_inventory = true,
    inventory_fns = [
        (d, s, o) -> s[Compound(:has, [Const(:human), o])],
        (d, s, o) -> s[Compound(:has, [Const(:robot), o])]
    ],
    inventory_types = [:item, :item],
    inventory_labels = ["Human Inventory", "Robot Inventory"],
    trajectory_options = Dict(
        :tracked_objects => [Const(:human), Const(:robot)],
        :tracked_types => Const[],
        :object_colors => [:black, :slategray]
    )
)

# Maps from assistance types to renderers
renderer_dict = Dict(
    "none" => RENDERER,
    "keys" => RENDERER_LABELED_KEYS,
    "doors" => RENDERER_LABELED_DOORS
)

"Adds a subplot to a storyboard with a line plot of goal probabilities."
function storyboard_goal_lines!(
    storyboard::Figure, goal_probs, ts=Int[];
    goal_names = ["(has gem1)", "(has gem2)", "(has gem3)", "(has gem4)"],
    goal_colors = PDDLViz.colorschemes[:vibrant][1:length(goal_names)],
    show_legend = false, ts_linewidth = 1, ts_fontsize = 24, 
    ax_args = (), kwargs...
)
    n_rows, n_cols = size(storyboard.layout)
    width, height = size(storyboard.scene)
    # Add goal probability subplot
    if length(ts) == size(goal_probs)[2]
        curves = [[Makie.Point2f(t, p) for (t, p) in zip(ts, goal_probs[i, :])]
                  for i in 1:size(goal_probs)[1]]
        ax, _ = series(
            storyboard[n_rows+1, 1:n_cols], curves;
            color = goal_colors, labels=goal_names,
            axis = (xlabel="Time", ylabel = "Probability",
                    limits=((1, ts[end]), (0, 1)), ax_args...),
            kwargs...
        )
    else
        ax, _ = series(
            storyboard[n_rows+1, 1:n_cols], goal_probs,
            color = goal_colors, labels=goal_names,
            axis = (xlabel="Time", ylabel = "Probability",
                    limits=((1, size(goal_probs, 2)), (0, 1)), ax_args...),
            kwargs...
        )
    end
    # Add legend to subplot
    if show_legend
        axislegend("Goals", framevisible=false)
    end
    # Add vertical lines at timesteps
    if !isempty(ts)
        vlines!(ax, ts, color=:black, linestyle=:dash,
                linewidth=ts_linewidth)
        positions = [(t + 0.1, 0.85) for t in ts]
        labels = ["t = $t" for t in ts]
        text!(ax, positions; text=labels, color = :black,
              fontsize=ts_fontsize)
    end
    # Resize figure to fit new plot
    rowsize!(storyboard.layout, n_rows+1, Auto(0.25))
    resize!(storyboard, (width, height * 1.3))
    return storyboard
end
