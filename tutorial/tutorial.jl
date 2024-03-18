#===============================================================================
# Mini-CLIPS: A conceptual introduction to Cooperative Language-Guided Inverse Plan Search (CLIPS)

This tutorial will introduce you to (a miniature version of) CLIPS, a Bayesian
framework for grounded instruction following and goal assistance that accounts
for pragmatic context. Using CLIPS, you can build AI assistants that:

- Infer a user's goals and commands from their actions and instructions.
- Reliably ground inferred commands into executable sequences of actions.
- Interpret ambiguous instructions by taking into account the user's goals and actions.
- Maintain uncertainty over the user's goals if there is insufficient information.
 
## Outline

1. [Environment setup](#setup)
2. [Defining a Bayesian user model](#defining-model)
3. [Inferring goals from user actions](#goals-from-actions)
4. [Modeling natural language instructions with LLMs](#modeling-instructions)
5. [Inferring goals from actions and instructions](#goals-from-instructions)
6. [Interactive user assistance](#user-assistance)
7. [Possible extensions](#extensions)

## 1. Environment setup <a name="setup"></a>

Since CLIPS is a Bayesian framework that uses large language models (LLMs) to
model how people communicate instructions in language, we're going to use the
[Gen.jl](https://www.gen.dev/) probabilistic programming system to implement
Bayesian models, and the [GenGPT3.jl](https://github.com/probcomp/GenGPT3.jl)
library to query OpenAI's LLMs for completions and their probabilities.
 
If you haven't installed Gen and GenGPT3.jl by following the setup instructions
for this repository, you can activate a new environment in the directory 
containing this notebook, install the packages using Julia's package manager:
===============================================================================#
## Uncomment to run
## using Pkg
## Pkg.add(["Gen", "IterTools"])
## Pkg.add(url="https://github.com/probcomp/GenGPT3.jl.git")
#===============================================================================
(If you've already followed the setup instructions, you can simply activate the
associated environnment with `Pkg.activate("@__DIR__/..")`.)

Now we can load Gen, GenGPT3.jl, and other utilities into our current workspace. 
(If this is the first time you're using these packages, precompiling them may
take some time.)
===============================================================================#
using Gen, GenGPT3
using Printf, Random, IterTools
#===============================================================================
Next, make sure your OpenAI API key is set as an environment variable. You can
do so by [following this guide](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety),
or by manually setting the value of `ENV["OPENAI_API_KEY"]` in the cell below.
However, it is *not* recommended to save your API key in this notebook.
===============================================================================#
## Uncomment to run
## ENV["OPENAI_API_KEY"] = "..."
#===============================================================================
## 2. Defining a Bayesian user model <a name="defining-model"></a>

Using Gen, we can define a Bayesian model of a user who interacts with the
environment in a structured way:
1. First, the user decides on a goal $g$.
2. Next, they come up with a plan (or policy) $\pi$ to achieve the goal $g$.
3. Then at each step $t$, the user eithers take an action $a_t$ by following the
   plan, or communicate parts of their plan as an instruction $u_t$.

Our AI assistant gets to observe $a_t$ and $u_t$ at each step $t$, and can
choose to take actions in response. As an example, let's consider a grocery
store as our environment, where the user's goal is to acquire ingredients for
one of four possible recipes:
===============================================================================#
## Set of possible goals
GOALS = [
    "greek_salad",
    "veggie_burger",
    "fried_rice",
    "burrito_bowl"
];
#===============================================================================
In the full version of CLIPS, goals are defined as conjunctions of predicates 
(i.e. facts about the environment) that the user wants to achieve. A planning
algorithm then *automatically* derives a plan or policy $\pi$ that the user 
might follow to their goal $g$. For simplicity, we'll instead *manually* define
a (partially ordered) plan to each goal:
===============================================================================#
## Plans to each goal
PLANS = Dict(
   "greek_salad" => Dict(
        "get(tomato)" => String[],
        "get(olives)" => String[],
        "get(cucumber)" => String[],
        "get(onion)" => String[],
        "get(feta_cheese)" => String[],
        "checkout()" => String[
            "get(tomato)", "get(olives)", "get(cucumber)",
            "get(onion)", "get(feta_cheese)"
        ]
   ),
   "veggie_burger" => Dict(
        "get(hamburger_bun)" => String[],
        "get(tomato)" => String[],
        "get(onion)" => String[],
        "get(lettuce)" => String[],
        "get(frozen_patty)" => String[
            "get(hamburger_bun)", "get(tomato)",
            "get(onion)", "get(lettuce)"
        ],
        "checkout()" => String[
            "get(hamburger_bun)", "get(tomato)", "get(onion)",
            "get(lettuce)", "get(frozen_patty)"
        ]
   ),
   "fried_rice" => Dict(
        "get(rice)" => String[],
        "get(onion)" => String[],
        "get(soy_sauce)" => String[],
        "get(frozen_peas)" => String[
            "get(rice)", "get(onion)", "get(soy_sauce)"
        ],
        "get(frozen_carrots)" => String[
            "get(rice)", "get(onion)", "get(soy_sauce)"
        ],
        "checkout()" => String[
            "get(rice)", "get(onion)", "get(soy_sauce)",
            "get(frozen_peas)", "get(frozen_carrots)"
        ]
   ),
   "burrito_bowl" => Dict(
        "get(rice)" => String[],
        "get(black_beans)" => String[],
        "get(cotija_cheese)" => String[],
        "get(onion)" => String[],
        "get(tomato)" => String[],
        "checkout()" => String[
            "get(rice)", "get(black_beans)", "get(cotija_cheese)",
            "get(onion)", "get(tomato)"
        ]
   )
);

## Set of possible actions
ACTIONS = sort!(collect(union((keys(plan) for plan in values(PLANS))...)))
push!(ACTIONS, "wait()");
#===============================================================================
Each partially ordered plan corresponds to a set of actions that must be
performed to achieve the goal, along with dependencies between those actions.
For example, the `checkout()` action has to be performed after all other
actions. We also assume that the user plans to collect frozen food items only
after acquiring all non-frozen items.

With our goals and plans defined, we can now model how a user takes actions 
by writing a probabilistic program:
===============================================================================#
"Labeled uniform distribution."
@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

"Labeled categorical distribution."
@dist labeled_categorical(labels, probs) = labels[categorical(probs)]

"Returns the actions the user might execute at a `state` given their `plan`."
function get_planned_actions(state::Set, plan::Dict)
    planned_acts = filter(collect(keys(plan))) do act
        if act in state 
            return false # Filter out completed actions
        elseif !all(act_dep in state for act_dep in plan[act])
            return false # Filter out actions with unfulfilled dependencies
        else
            return true # Keep actions that are planned and not completed
        end
    end
    sort!(planned_acts)
    if isempty(planned_acts)
        push!(planned_acts, "wait()")
    end
    return planned_acts
end

"Model of user's goal-directed actions over time."
@gen function act_only_user_model(T::Int, act_noise::Real = 0.05)
    ## Construct initial state of environment
    state = Set{String}()
    ## Sample user's goal and select plan
    goal ~ labeled_uniform(GOALS)
    plan = PLANS[goal]
    ## Sample actions at each timestep
    act_history = String[]
    for t in 1:T
        ## Determine user's next possible actions
        planned_acts = get_planned_actions(state, plan)
        planned_probs = fill((1.0 - act_noise) / length(planned_acts),
                             length(planned_acts))
        ## Determine set of unexecuted actions
        possible_acts = filter(!in(state), ACTIONS)
        possible_probs = fill(act_noise / length(possible_acts),
                              length(possible_acts))
        ## Sample next action (with some action noise)
        next_acts = vcat(planned_acts, possible_acts)
        next_act_probs = vcat(planned_probs, possible_probs)
        act = {(:act, t)} ~ labeled_categorical(next_acts, next_act_probs)
        ## Update state and action history
        if act != "wait()"
            push!(state, act)
        end
        push!(act_history, act)
    end
    ## Return final state and action history
    return (state, act_history)
end
#===============================================================================
In this program, we first sample the user's `goal` from a uniform prior over
goals. We then select the `plan` that corresponds to that goal, and 
simulate how the user might act in accordance with that plan for `T` steps.

Specifically, we assume that with probability `(1 - act_noise)`, the user
selects one of their planned actions. Otherwise, the user makes a "mistake",
executing a random action. After taking an action, we update the environment
`state` to keep track of all actions that have been achieved.

Let's sample an execution trace from this model, and see what it looks like:
===============================================================================#
T, act_noise = 6, 0.0
trace = Gen.simulate(act_only_user_model, (T, act_noise))
goal = trace[:goal]
state, act_history = get_retval(trace)
@show goal;
@show act_history;
@show state;
#===============================================================================
As you can see, the simulated actions correspond with the sampled goal. The 
order of the actions also respects the constraints imposed by the partially
ordered plan.

With Gen, we aren't limited to simulating possible traces from a probabilistic
program. We can also *evaluate* the probability that a particular trace was 
generated by our program. We do this by specifying a **choice map**: a mapping
of random variables to specific values:
===============================================================================#
## Specify values of random choices
constraints = choicemap(
    (:goal, "greek_salad"),
    ((:act, 1), "get(tomato)"),
    ((:act, 2), "get(onion)"),
    ((:act, 3), "get(olives)"),
)

## Evaluate the probability of a trace with those choices
T, act_noise = 3, 0.0
trace, _ = Gen.generate(act_only_user_model, (T, act_noise), constraints)
p_trace = exp(Gen.get_score(trace))

## P(trace) = P(goal) P(actions | goal) = 1/4 * (1/5 * 1/4 * 1/3) ≈ 0.004167
@show p_trace;
#==============================================================================
If you manually work out the probability of the trace, you'll see that it's 
equal to the value that Gen automatically calculated for us.

## 3. Inferring goals from user actions <a name="goals-from-actions"></a>

Since we can evaluate the probability of each trace, we can use this to
implement an enumerative Bayesian inference algorithm:
===============================================================================#
"""
    enum_inference(model, model_args, observations, latent_addrs, latent_values)

Runs enumerative Bayesian inference for a `model` parameterized by `model_args`, 
conditioned on the `observations`. Given a list of `latent_addrs` and the 
a list of corresponding `latent_values` that each latent variable can take on,
we enumerate over all possible settings of the latent variables.

Returns a named tuple with the following fields:
- `traces`: An array of execution traces for each combination of latent values.
- `logprobs`: An array of log probabilities for each trace.
- `latent_logprobs`: A dictionary of log posterior probabilities per latent.
- `latent_probs`: A dictionary of posterior probabilities per latent.
- `lml`: The log marginal likelihood of the observations.
"""
function enum_inference(
    model::GenerativeFunction, model_args::Tuple,
    observations::ChoiceMap, latent_addrs, latent_values
) 
    @assert length(latent_addrs) == length(latent_values)
    ## Construct iterator over combinations of latent values
    latents_iter = Iterators.product(latent_values...)
    ## Generate a trace for each possible combination of latent values
    traces = map(latents_iter) do latents
        constraints = choicemap()
        for (addr, val) in zip(latent_addrs, latents)
            constraints[addr] = val
        end
        constraints = merge(constraints, observations)
        tr, _ = Gen.generate(model, model_args, constraints)
        return tr
    end
    ## Compute the log probability of each trace
    logprobs = map(Gen.get_score, traces)
    ## Compute the log marginal likelihood of the observations
    lml = logsumexp(logprobs)
    ## Compute the (marginal) posterior probabilities for each latent variable
    latent_logprobs = Dict(
        addr => ([logsumexp(lps) for lps in eachslice(logprobs, dims=i)] .- lml)
        for (i, addr) in enumerate(latent_addrs)
    )
    latent_probs = Dict(addr => exp.(lp) for (addr, lp) in latent_logprobs)
    return (
        traces = traces,
        logprobs = logprobs,
        latent_logprobs = latent_logprobs,
        latent_probs = latent_probs,
        latent_addrs = latent_addrs,
        lml = lml
    )
end
#===============================================================================
By running this algorithm on a sequence of observed actions $a_{1:t}$, we can
infer the goal posterior $P(g | a_{1:t})$ given a sequence of actions:

$$P(g | a_{1:t}) = \frac{P(g, a_{1:t})}{P(a_{1:t})}$$

Let's see what happens when we observe that the user gets a tomato, followed
by an onion:
===============================================================================#
## Observed actions
observations = choicemap(
    ((:act, 1), "get(tomato)"),
    ((:act, 2), "get(onion)"),
)

## Run inference by enumerating over all possible goals
T = 2
results = enum_inference(
    act_only_user_model, (T,), observations, (:goal,), (GOALS,)
)

## Show inferred goal probabilities
println("P(goal)\t\tgoal")
for (goal, prob) in zip(GOALS, results.latent_probs[:goal])
    @printf("%.3f\t\t%s\n", prob, goal)
end
#===============================================================================
The inferred distribution shows that user the might be collecting ingredients
for Greek salad, a veggie burger, or a burrito_bowl. This makes sense, since
only those recipes require both an onion and tomato. In contrast, fried rice
is very unlikely to be the user's goal.

Interestingly, the veggie burger is more likely than the other two possible
goals. This is because there are *more possible ways* to get the 5 ingredients
required for the Greek salad or burrito bowl, whereas the user has to collect
4 non-frozen ingredients for the veggie burger before obtaining the frozen
veggie patty. As a result, if the user's goal was Greek salad, it's slightly
less likely that they would happen to pick up the onion and tomato. After all,
they could have easily picked up the olives or cucumber first!

Now let's see what happens if the user get some rice after collecting the
tomato and onion:
===============================================================================#
observations = choicemap(
    ((:act, 1), "get(tomato)"),
    ((:act, 2), "get(onion)"),
    ((:act, 3), "get(rice)"),
)

T = 3
results = enum_inference(
    act_only_user_model, (T,), observations, (:goal,), (GOALS,)
)

println("P(goal)\t\tgoal")
for (goal, prob) in zip(GOALS, results.latent_probs[:goal])
    @printf("%.3f\t\t%s\n", prob, goal)
end
#===============================================================================
Once the user adds some rice to their shopping cart, it becomes obvious
that they are shopping for a burrito bowl (assuming there are no other possible 
goals). The posterior over goals reflects this certainty.

## 4. Modeling natural language instructions with LLMs <a name="modeling-instructions"></a>

At this point, we've built a model of how user acts over time to achieve their
goal, then shown you how to infer the user's goal given a series of actions. Now
we'll extend this with an *utterance model*, which models how a user might 
communicate their plan $\pi$ at state $s_t$ as an instruction in
natural language:
===============================================================================#
## Define LLM mixture-of-prompts model
gpt3_mixture =
    GPT3Mixture(model="davinci-002", stop="\n", max_tokens=512)

## Few-shot examples of how commands are translated into instructions
COMMAND_EXAMPLES = [
    ("get(apple)", "Can you get the apple?"),
    ("get(bread)", "Could you find some bread?"),
    ("get(cheddar_cheese)", "Go grab a block of that cheese."),
    ("get(green_tea)", "Add some tea to the cart."),
    ("checkout()", "Let's checkout."),
    ("get(tofu) get(seitan)", "I need some tofu and seitan."),
    ("get(frozen_mango) get(ice_cream)", "Get the mango and ice cream."),
    ("get(strawberries) get(milk)", "Find me strawberries and milk."),
    ("get(frozen_broccoli) get(frozen_cauliflower)", "We'll need frozen broccoli and cauliflower."),
    ("get(fries) checkout()", "Let's get some fries then checkout."),
]
Random.seed!(0)
shuffle!(COMMAND_EXAMPLES)

"Construct few-shot prompt for translating a command into an instruction."
function construct_utterance_prompt(
    command::Vector{String}, examples = COMMAND_EXAMPLES
)
    example_strs = ["Input: $cmd\nOutput: $utt" for (cmd, utt) in examples]
    example_str = join(example_strs, "\n")
    command_str = join(command, " ")
    prompt = "$example_str\nInput: $command_str\nOutput:"
    return prompt
end

"Returns future planned actions in topologically sorted order."
function get_future_actions(state::Set, plan::Dict)
    future_acts = String[]
    visited = Set{String}()
    finished = Set{String}()
    queue = collect(keys(plan))
    while !isempty(queue)
        act = queue[end]
        if act in finished
            pop!(queue)
            continue
        elseif act in visited
            pop!(queue)
            push!(finished, act)
            act in state || push!(future_acts, act)
        else
            push!(visited, act)
            for act_dep in plan[act]
                act_dep in finished && continue
                act_dep in visited && error("Cycle detected!")
                push!(queue, act_dep)
            end
        end
    end
    return future_acts
end

"Model of how utterances are generated given the user's goal and plan."
@gen function utterance_model(goal::String, plan::Dict, state::Set)
    ## Determine the set of future planned actions
    future_acts = get_future_actions(state, plan)
    ## Enumerate all subsets of up to two actions as commands
    commands = Vector{String}[]
    for k in 1:2, acts in IterTools.subsets(future_acts, k)
        push!(commands, collect(acts))
    end
    ## Construct a prompt for each possible command
    prompts = [construct_utterance_prompt(cmd) for cmd in commands]
    ## Generate an utterance from the LLM mixture-of-prompts model
    utterance ~ gpt3_mixture(prompts)
    return (utterance, commands)
end
#===============================================================================
Our utterance model is a model of *goal-directed communication*: Given some plan
$\pi$ that the user has in mind, we:
1. Select a random subset of the actions that have yet to be taken, forming
   a *command* $c_t$: a code-like specification of the user's instruction.
2. Translate this command into a natural language instruction $u_t$ using an LLM
   (in this case, OpenAI's `davinci-002` model).

How does the LLM perform this translation? Since LLMs are capable of in-context
learning, all we have to provide is a list of few-shot examples
(`COMMAND_EXAMPLES`) in the LLM's prompt. We then append the command we want
to translate to the end of the prompt (`construct_utterance_prompt`),
and generate a completion from the LLM.

<details>
<summary>Tell me more: Variance reduction via command enumeration.</summary><br>

One thing you might notice about the utterance model is that we aren't actually
sampling a random command $c_t$ to translate. Instead, we're enumerating over
all possible commands, constructing a prompt for each, then passing them to a 
*mixture-of-prompts* model (`gpt3_mixture`). Under-the-hood, this mixture model
samples a random prompt from the list of prompts, then generates a completion
for that prompt. It then evaluates the total probability of the completion under
*all* possible prompts.

Why go through this process of enumerating all commands and their corresponding
prompts, especially when it requires more calls to the LLM? The main reason
is *variance reduction*: By using the mixture model, we can directly evaluate
the likelihood $P(u_t | \pi, g)$ of an utterance $u_t$ given the user's plan
$\pi$ and goal $g$. If we instead sampled a command $c_t ~ P(c_t | \pi, g)$, we
would get a noisy estimate $P(u_t| c_t, \pi, g) \approx P(u_t | \pi, g)$ instead
of the exact value. When the space of commands is small enough to enumerate
over, this reduction in variance can often be worthwhile.
</details><br>

Let's see what utterance gets generated by our model when we specify the user's
goal and plan:
===============================================================================#
goal = "fried_rice"
plan = PLANS[goal]
state = Set{String}()
utterance, commands = utterance_model(goal, plan, state)
@show utterance;
#===============================================================================
Of course, we don't just want to generate instructions from an LLM. We want to 
*observe* them, then evaluate how likely that instruction is given the user's
goal and plan. We can do this using the `Gen.generate` function we saw earlier:
===============================================================================#
## Specify utterance (with starting space to match OpenAI tokenization)
utterance = " We need soy sauce and onions."
observations = choicemap((:utterance => :output, utterance))

## Evaluate the log probability of the utterance
trace, _ = Gen.generate(utterance_model, (goal, plan, state), observations)
likely_utterance_logprob = Gen.get_score(trace)
@show likely_utterance_logprob;
#===============================================================================
We can also display the (local) posterior distribution over commands
$P(c_t | u_t, \pi, g)$ that might have generated the observed instruction $u_t$:
===============================================================================#
## Show most likely commands, given the observed utterance
commands = Gen.get_retval(trace)[2]
command_probs = trace[:utterance => :post_probs]
top_5_idxs = sortperm(command_probs, rev=true)[1:5]
println("P(command)\tcommand")
for idx in top_5_idxs
    @printf("%.3f\t\t%s\n", command_probs[idx], commands[idx])
end
#===============================================================================
What happens if we observe an utterance that is very unlikely given the user's
plan (e.g. `" Let's get some soap."`)? We should then expect the probability 
of the utterance $P(u_t | \pi, g)$ to be lower. Furthermore, since no command 
$c_t$ can explain the utterance well, we should see a much more uncertain
distribution $P(c_t | u_t, \pi, g)$ over the possible commands: 
===============================================================================#
## Specify unlikely utterance (with starting space to match OpenAI tokenization)
utterance = " Let's get some soap."
observations = choicemap((:utterance => :output, utterance))

## Evaluate the log probability of the unlikely utterance
trace, _ = Gen.generate(utterance_model, (goal, plan, state), observations)
unlikely_utterance_logprob = Gen.get_score(trace)
@show unlikely_utterance_logprob

## Show most likely commands, given the observed utterance
commands = Gen.get_retval(trace)[2]
command_probs = trace[:utterance => :post_probs]
top_5_idxs = sortperm(command_probs, rev=true)[1:5]
println("P(command)\tcommand")
for idx in top_5_idxs
    @printf("%.3f\t\t%s\n", command_probs[idx], commands[idx])
end
#===============================================================================
## 5. Inferring goals from actions and instructions <a name="goals-from-instructions"></a>

So far we've seen how to evaluate the likelihood $P(u_t | \pi, g)$ of an
utterance $u_t$ given a known plan $\pi$ and goal $g$, and to compute the
*local* posterior distribution over commands $P(c_t | u_t, \pi, g)$.

In general, however, we don't know the user's goal $g$ or plan $\pi$. To 
model this situation, we need to embed `utterance_model` as a *sub-routine*
within our full user model:
===============================================================================#
"Model of user's goal-directed actions and instructions over time."
@gen function full_user_model(T::Int, act_noise::Real = 0.05)
    ## Construct initial state of environment
    state = Set{String}()
    ## Sample user's goal and select plan
    goal ~ labeled_uniform(GOALS)
    plan = PLANS[goal]
    ## Decide whether to speak at the beginning
    speak = {(:speak, 0)} ~ bernoulli(0.2)
    ## Generate utterance that communicates the current goal and plan
    if speak
        {(:utterance, 0)} ~ utterance_model(goal, plan, state)
    end
    ## Sample actions and utterances at each timestep
    act_history = String[]
    for t in 1:T
        ## Determine user's next possible actions
        planned_acts = get_planned_actions(state, plan)
        planned_probs = fill((1.0 - act_noise) / length(planned_acts),
                             length(planned_acts))
        ## Determine set of unexecuted actions
        possible_acts = filter(!in(state), ACTIONS)
        possible_probs = fill(act_noise / length(possible_acts),
                              length(possible_acts))
        ## Sample next action (with some action noise)
        next_acts = vcat(planned_acts, possible_acts)
        next_act_probs = vcat(planned_probs, possible_probs)
        act = {(:act, t)} ~ labeled_categorical(next_acts, next_act_probs)
        ## Update state and action history
        if act != "wait()"
            push!(state, act)
        end
        push!(act_history, act)
        ## Decide whether to speak at this timestep
        speak = {(:speak, t)} ~ bernoulli(0.2)
        ## Generate utterance that communicates the current goal and plan
        if speak
            {(:utterance, t)} ~ utterance_model(goal, plan, state)
        end
    end
    ## Return final state and action history
    return (state, act_history)
end
#===============================================================================
This model extends `act_only_user_model` by possibly generating an utterance at 
each step $t$. This is done by first sampling whether or not the user speaks
(denoted by the address `(:speak, t)`), then generating an utterance from the 
`utterance_model` if the user decides to speak.

Since this model describes how a user might act and talk given a particular
goal, we can condition on both actions and instructions to infer their goal:

**Note: Latency of the following code may be high since we are making multiple
requests to OpenAI's LLM API over the web, instead of running an LLM locally.**
===============================================================================#
## Observed actions and instructions
observations = choicemap(
    ((:speak, 0), true),
    ((:utterance, 0) => :utterance => :output, " Can you grab a tomato?"),
    ((:act, 1), "get(onion)"), ((:speak, 1), false),
)

## Run inference by enumerating over all possible goals
T = 1
results = enum_inference(
    full_user_model, (T,), observations, (:goal,), (GOALS,)
)

## Show inferred goal probabilities
println("P(goal)\t\tgoal")
for (goal, prob) in zip(GOALS, results.latent_probs[:goal])
    @printf("%.3f\t\t%s\n", prob, goal)
end
#===============================================================================
In the example above, we first observe the user say `" Can you grab a tomato?"` 
at $t=0$. At $t=1$, the user then takes the action `get(onion)`. Bayesian 
inference automatically combines these pieces of information, inferring once 
again that the user might be shopping for a Greek salad, veggier burger, or
burrito bowl.

In addition to inferring the user's goal $g$, we can infer the command $c_t$
that led to the instruction $u_t$. We do this by summing the local command
posteriors $P(c_t | u_t, \pi, g)$ across possible plans $\pi$ and goals $g$:
===============================================================================#
"Extract posterior over commands at step `t` given a list of weighted traces."
function extract_command_probs(
    t::Int, traces::AbstractVector{<:Trace}, logprobs::AbstractVector{<:Real}
)
    ## Sum local command posteriors across traces
    command_probs = Dict{Vector{String}, Float64}()
    log_total = logsumexp(logprobs)
    for (tr, lp) in zip(traces, logprobs)
        _, trace_commands = tr[(:utterance, t)]
        trace_command_probs = tr[(:utterance, t) => :utterance => :post_probs]
        trace_prob = exp(lp - log_total)
        for (cmd, p) in zip(trace_commands, trace_command_probs)
            command_probs[cmd] = get(command_probs, cmd, 0.0) + p * trace_prob
        end
    end
    ## Sort commands by probability
    commands = collect(keys(command_probs))
    command_probs = collect(values(command_probs))
    idxs = sortperm(command_probs, rev=true)
    return commands[idxs], command_probs[idxs]
end

## Show top inferred command probabilities
commands, command_probs = extract_command_probs(0, results.traces, results.logprobs)
println("P(command)\tcommand")
for (cmd, prob) in zip(commands[1:5], command_probs[1:5])
    @printf("%.3f\t\t%s\n", prob, cmd)
end
#===============================================================================
As expected, the utterance `" Can you grab a tomato?"` is most likely to have
been generated from the command `get(tomato)`.

The ability to combine goal-relevant information across modalities means that
CLIPS can *disambiguate* instructions that would be ambiguous without context.
Imagine that the user says `"Can you get the stuff in the frozen section?"`
at $t=0$:
===============================================================================#
observations = choicemap(
    ((:speak, 0), true),
    ((:utterance, 0) => :utterance => :output,
     " Can you get the stuff in the frozen section?"),
)

T = 0
results = enum_inference(
    full_user_model, (T,), observations, (:goal,), (GOALS,)
)

println("P(goal)\t\tgoal")
for (goal, prob) in zip(GOALS, results.latent_probs[:goal])
    @printf("%.3f\t\t%s\n", prob, goal)
end
println()

commands, command_probs = extract_command_probs(0, results.traces, results.logprobs)
println("P(command)\tcommand")
for (cmd, prob) in zip(commands[1:5], command_probs[1:5])
    @printf("%.3f\t\t%s\n", prob, cmd)
end
#===============================================================================
Given the limited information, it's not clear if the user wants the frozen
patty (for the veggie burger), or the frozen peas and carrots (for fried rice).
As a result, CLIPS assigns about equal probability to each of those goals.

However, if we see the user get some rice before asking for the frozen stuff,
their intentions become much clearer:
===============================================================================#
observations = choicemap(
    ((:speak, 0), false),
    ((:act, 1), "get(rice)"), ((:speak, 1), true),
    ((:utterance, 1) => :utterance => :output,
     " Can you get the stuff in the frozen section?"),
)

T = 1
results = enum_inference(
    full_user_model, (T,), observations, (:goal,), (GOALS,)
)

println("P(goal)\t\tgoal")
for (goal, prob) in zip(GOALS, results.latent_probs[:goal])
    @printf("%.3f\t\t%s\n", prob, goal)
end
println()

commands, command_probs = extract_command_probs(1, results.traces, results.logprobs)
println("P(command)\tcommand")
for (cmd, prob) in zip(commands[1:5], command_probs[1:5])
    @printf("%.3f\t\t%s\n", prob, cmd)
end
#===============================================================================
Try modifying the code above to see how CLIPS handles ambiguous instructions
in the presence or absence of actions. Here are some possibilities:
- `" I need to get some veggies."`
- `" Could you help find the cheese?"`
- `" Let's get the last ingredient then checkout."`
- `" We need to get the fresh stuff first."`

For each of these cases, the instruction alone is not enough to determine the
user's true goal. Actions are required to disambiguate the user's intentions.

## 6. Interactive user assistance <a name="user-assistance"></a>

We've now seen how to model a user's actions and instructions over time, and how
to infer their goal and commands given those actions and instructions. All of
this is still *passive*, however. In this section, we'll show you how to write
an *interactive assistant* that takes actions based on what it infers about 
the user at each step $t$.

To do this, we'll need to adjust our user model so that it takes a list of 
*assistant actions* as an argument. At each step $t$, the assistant either 
does nothing or takes an action. If the assistant does nothing, then the user 
will act; otherwise, only the assistant will act. The user may also give 
instructions at any step:
===============================================================================#
"Model of a user interacting with an assistant over time."
@gen function interactive_user_model(
    assist_actions::AbstractVector, act_noise::Real = 0.05
)
    ## Construct initial state of environment
    state = Set{String}()
    ## Sample user's goal and select plan
    goal ~ labeled_uniform(GOALS)
    plan = PLANS[goal]
    ## Decide whether to speak at the beginning
    speak = {(:speak, 0)} ~ bernoulli(0.2)
    ## Generate utterance that communicates the current goal and plan
    if speak
        {(:utterance, 0)} ~ utterance_model(goal, plan, state)
    end
    ## Sample actions and utterances at each timestep
    act_history = String[]
    for (t, assist_act) in enumerate(assist_actions)
        if isnothing(assist_act) # User acts, assistant does not
            ## Determine user's next possible actions
            planned_acts = get_planned_actions(state, plan)
            planned_probs = fill((1.0 - act_noise) / length(planned_acts),
                                 length(planned_acts))
            ## Determine set of unexecuted actions
            possible_acts = filter(!in(state), ACTIONS)
            possible_probs = fill(act_noise / length(possible_acts),
                                  length(possible_acts))
            ## Construct action distribution
            next_acts = vcat(planned_acts, possible_acts)
            next_act_probs = vcat(planned_probs, possible_probs)
        else # Assistant acts, user does not
            @assert assist_act in ACTIONS
            next_acts = [assist_act]
            next_act_probs = [1.0]
        end
        ## Sample next action
        act = {(:act, t)} ~ labeled_categorical(next_acts, next_act_probs)
        ## Update state and action history
        if act != "wait()"
            push!(state, act)
        end
        push!(act_history, act)
        ## Decide whether to speak at this timestep
        speak = {(:speak, t)} ~ bernoulli(0.2)
        ## Generate utterance that communicates the current goal and plan
        if speak
            {(:utterance, t)} ~ utterance_model(goal, plan, state)
        end
    end
    ## Return final state and action history
    return (state, act_history)
end
#===============================================================================
We'll also extend our enumerative inference algorithm so that we can update 
our inferences step-by-step, instead of having to observe the user's actions 
all at once:
===============================================================================#
"""
    enum_inference_step(prev_results, new_model_args, new_observations)

Updates a set of inference results (`prev_results`) by adusting the model's
arguments to `new_model_args`, and conditioning on `new_observations`.
"""
function enum_inference_step(
    prev_results::NamedTuple, new_model_args::Tuple, new_observations::ChoiceMap
)
    ## Update previous traces with the new arguments and observations
    argdiffs = map(_ -> UnknownChange(), new_model_args)
    traces = map(prev_results.traces) do prev_trace
        trace, _, _, _ =
            Gen.update(prev_trace, new_model_args, argdiffs, new_observations)
        return trace
    end
    ## Compute the log probability of each trace
    logprobs = map(Gen.get_score, traces)
    ## Compute the log marginal likelihood of the observations
    lml = logsumexp(logprobs)
    ## Compute the (marginal) posterior probabilities for each latent variable
    latent_logprobs = Dict(
        addr => ([logsumexp(lps) for lps in eachslice(logprobs, dims=i)] .- lml)
        for (i, addr) in enumerate(prev_results.latent_addrs)
    )
    latent_probs = Dict(addr => exp.(lp) for (addr, lp) in latent_logprobs)
    return (
        traces = traces,
        logprobs = logprobs,
        latent_logprobs = latent_logprobs,
        latent_probs = latent_probs,
        latent_addrs = prev_results.latent_addrs,
        lml = lml
    )
end
#===============================================================================
Finally, we'll define an assistance policy that takes in the observation history
and inferences about the user, then returns an action. We'll take a relatively
conservative approach where:

- The assistant only acts after the user has spoken / given an instruction.
- If the user's goal or command is too uncertain, the assistant does nothing.
- Otherwise, the assistant follows the inferred command by taking 
  the most likely uncompleted action that is part of the command.
- If all commanded actions are completed, the assistant does nothing.
===============================================================================#
"""
    assistance_policy(t, state, observations, results)

Returns an assistive action given the current step `t`, environment `state`,
history of `observations`, and inferences about the user (`results`).
"""
function assistance_policy(
    t::Int, state::Set, observations::ChoiceMap, results::NamedTuple;
    goal_thresh::Real = 0.25, cmd_thresh::Real = 0.5
)
    ## Find most recent instruction
    t_speak = nothing
    for i in t:-1:0
        Gen.has_value(observations, (:speak, i)) || continue
        observations[(:speak, i)] == true || continue
        t_speak = i
        break
    end
    ## Do nothing if user has not spoken
    isnothing(t_speak) && return nothing
    ## Extract current posterior over goals and most recent command
    goal_probs = results.latent_probs[:goal]
    commands, command_probs =
        extract_command_probs(t_speak, results.traces, results.logprobs)
    ## Determine most likely action from command distribution
    act_probs = Dict{String,Float64}()
    for (cmd, prob) in zip(commands, command_probs), act in cmd
        act in state && continue # Ignore completed actions
        act_probs[act] = get!(act_probs, act, 0.0) + prob
    end
    isempty(act_probs) && return nothing
    max_act_prob, max_act = findmax(act_probs)
    ## Do nothing if user's goal or command is too uncertain
    maximum(goal_probs) < goal_thresh && return nothing
    max_act_prob < cmd_thresh && return nothing
    ## Take the most likely uncompleted action
    return max_act
end
#===============================================================================
Note that this is only one way to design an assistance policy. In the full
version of CLIPS, we implemented a more pro-active form of assistance which acts
at every step to minimize the expected cost of achieving the user's goal.
While this can be more helpful, it can also provide help even in cases where the
user has not requested for help, and may not even want it. The policy shown
above avoids such situations.

Now let's put everything together into a Read-Evaluate-Print Loop (REPL). To 
make things more fun, we'll turn this into a game, where:
- You, the user, will be given a goal to achieve. 
- At each step, you can either act or give an instruction to the assistant.
- The assistant will respond if it's confident enough about your intentions. 
===============================================================================#
"REPL-based assistance game with a CLIPS assistance policy."
function clips_repl(
    max_steps::Int = 10;
    act_noise::Real = 0.05, goal_thresh::Real = 0.25, cmd_thresh::Real = 0.5
)
    assist_actions = Union{String, Nothing}[]
    act_history = String[]
    state = Set{String}()
    ## Sample a random goal for the user to achieve
    true_goal = rand(GOALS)
    ## Construct initial observation choicemap (no speaking at t = 0)
    utterance = ""
    observations = choicemap(((:speak, 0), false))
    ## Initialize goal and command inferences
    results = enum_inference(
        interactive_user_model, (assist_actions, act_noise), observations,
        (:goal,), (GOALS,)
    )
    ## Loop up to maximum number of steps
    for t in 1:max_steps
        if !isempty(assist_actions) && isnothing(assist_actions[end])
            ## Show inferred goal probabilities
            println("P(goal)\t\tgoal")
            println("-"^50)
            for (goal, prob) in zip(GOALS, results.latent_probs[:goal])
                @printf("%.3f\t\t%s\n", prob, goal)
            end
            println()
            ## Show inferred command probabilities
            if !isempty(utterance)
                commands, command_probs =
                    extract_command_probs(t-1, results.traces, results.logprobs)
                println("P(command)\tcommand")
                println("-"^80)
                for (cmd, prob) in zip(commands[1:5], command_probs[1:5])
                    @printf("%.3f\t\t%s\n", prob, cmd)
                end
                println()
            end
        end
        ## Decide whether to act based on assistance policy
        assist_act = assistance_policy(t-1, state, observations, results;
                                       goal_thresh, cmd_thresh)
        push!(assist_actions, assist_act)
        println("=== t = $t ===")
        ## Request user's next input if assistant does not act
        if isnothing(assist_act)
            possible_acts = filter(!in(state), ACTIONS)
            remaining_acts = get_future_actions(state, PLANS[true_goal])
            println("Goal: $true_goal", "\n")
            println("Remaining Actions: ", join(remaining_acts, " "), "\n")
            println("Possible Actions: ", join(possible_acts, " "), "\n")
            print("User: ")
            input = strip(readline())
            if input in possible_acts # User takes an action
                act = input
                utterance = ""
            else # User gives an instruction
                act = "wait()"
                utterance = input
            end
        else
            act = assist_act
            utterance = ""
            println("Assistant: ", act)
        end
        ## Update environment state
        push!(act_history, act)
        if act != "wait()"
            push!(state, act)
        end
        ## Check if the goal has been achieved
        if keys(PLANS[true_goal]) ⊆ state
            println("\n=== Goal achieved: $true_goal ===\n")
            return (state, act_history, assist_actions)
        end
        ## Construct next observation choicemap
        if !isempty(utterance)
            new_obs = choicemap(
                ((:speak, t), true), ((:act, t), act),
                ((:utterance, t) => :utterance => :output, " " * utterance),
            )
        else
            new_obs = choicemap(((:speak, t), false), ((:act, t), act))
        end
        observations = merge(observations, new_obs)
        ## Update goal and command inferences
        if !isempty(utterance)
            println("\nRunning inference (LLM queries may take a while)...\n")
        elseif isnothing(assist_act)
            println("\nRunning inference...\n")
        else
            println()
        end
        new_args = (assist_actions, act_noise)
        results = enum_inference_step(results, new_args, new_obs)
    end
    return (state, act_history, assist_actions)
end
#===============================================================================
We can now interact with the CLIPS assistance policy via the REPL:
===============================================================================#
(state, act_history, assist_actions) = clips_repl(10)
#===============================================================================
As you can see, the CLIPS assistant is quite accurate in its ability to respond
to ambiguous instructions, while still avoiding actions when the instructions
are very unclear. CLIPS also flexibly handles user input, whether it takes 
the form of actions or free-form text. This can easily be extended to a non-REPL
context, where users' take actions via UI elements in an app or video game.

## 7. Possible extensions <a name="extensions"></a>

Since the design of CLIPS is highly modular and interpretable, it's not hard to
add new features by enriching either the user model, the utterance model, or the
assistance policy. For example, you could expand the user model with:
- A larger space of possible goals, perhaps by sampling from an LLM.
- [Automatic plan generation](https://github.com/JuliaPlanners/SymbolicPlanners.jl) given a goal (as in the full version of CLIPS).

The utterance model could also support more types of utterances, such as:
- Utterances which directly inform the assistant about the goal.
- Instructions which refer back to an earlier instruction or assistant's action.
- Explicit modeling of outlier or adversarial instructions to improve robustness.

The assistance policy could be extended by:
- More pro-active forms of assistance (as in the full version of CLIPS).
- Asking of clarification questions in response to unclear instructions.

Finally, to reduce LLM usage and scale CLIPS to larger domains, some
possibilities include:
- Using smaller local LLMs which are fine-tuned on domain-relevant instructions.
- When translating commands to utterances, automatically retrieving similar
  few-shot examples to include in the LLM prompt.
- Instead of enumerating over all possible commands given a plan, inferring
  commands bottom-up from the utterance $u_t$, using constrained generation
  methods such as [Sequential Monte Carlo (SMC) steering](https://github.com/probcomp/hfppl)

That's it for this tutorial — happy experimenting with CLIPS!
===============================================================================#