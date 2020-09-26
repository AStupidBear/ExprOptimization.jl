
module GeneticPrograms

using StatsBase, Random
using BSON, ProgressMeter
using ExprRules, GCMAES

using ExprOptimization: ExprOptAlgorithm, ExprOptResult, BoundedPriorityQueue, enqueue!
import ExprOptimization: optimize

export GeneticProgram

const OPERATORS = [:reproduction, :crossover, :subtree_mutation, :hoist_mutation, :point_mutation]

abstract type InitializationMethod end 
abstract type SelectionMethod end
abstract type TrackingMethod end

"""
    GeneticProgram

Genetic Programming.
# Arguments
- `pop_size::Int`: population size
- `iterations::Int`: number of iterations
- `max_depth::Int`: maximum depth of derivation tree
- `p_reproduction::Float64`: probability of reproduction operator
- `p_crossover::Float64`: probability of crossover operator
- `p_mutation::Float64`: probability of mutation operator
- `init_method::InitializationMethod`: initialization method
- `select_method::SelectionMethod`: selection method
- `track_method::TrackingMethod`: additional tracking, e.g., track top k exprs (default: no additional tracking) 
"""
struct GeneticProgram <: ExprOptAlgorithm
    pop_size::Int
    iterations::Int
    max_depth::Int
    parsimony_coefficient::Float64
    p_point_mutation::Float64
    p_operators::Weights
    init_method::InitializationMethod
    select_method::SelectionMethod
    track_method::TrackingMethod

    function GeneticProgram(
        pop_size::Int,                          #population size 
        iterations::Int,                        #number of generations 
        max_depth::Int,                         #maximum depth of derivation tree
        parsimony_coefficient::Float64,         #parsimony coefficient
        p_reproduction::Float64,                #probability of reproduction operator 
        p_crossover::Float64,                   #probability of crossover operator
        p_subtree_mutation::Float64,            #probability of subtree mutation operator 
        p_hoist_mutation::Float64,              #probability of hoist mutation operator 
        p_point_mutation::Float64,              #probability of point mutation operator 
        p_point_replace::Float64;               #probability of point replace 
        init_method::InitializationMethod=RandomInit(),      #initialization method 
        select_method::SelectionMethod=TournamentSelection(),   #selection method 
        track_method::TrackingMethod=NoTracking())   #tracking method 

        p_operators = Weights([p_reproduction, p_crossover, p_subtree_mutation, p_hoist_mutation])
        new(pop_size, iterations, max_depth, parsimony_coefficient, p_point_mutation, p_operators, init_method, select_method, track_method)
    end
end

"""
    RandomInit

Uniformly random initialization method.
"""
struct RandomInit <: InitializationMethod end

"""
    TournamentSelection

Tournament selection method with tournament size k.
"""
struct TournamentSelection <: SelectionMethod 
    k::Int
end
TournamentSelection() = TournamentSelection(2)

"""
    TruncationSelection

Truncation selection method keeping the top k individuals 
"""
struct TruncationSelection <: SelectionMethod 
    k::Int 
end
TruncationSelection() = TruncationSelection(100)

"""
    NoTracking

No additional tracking of expressions.
"""
struct NoTracking <: TrackingMethod end

"""
    TopKTracking

Track the top k expressions.
"""
struct TopKTracking <: TrackingMethod 
    k::Int
    q::BoundedPriorityQueue{RuleNode,Float64}

    function TopKTracking(k::Int)
        q = BoundedPriorityQueue{RuleNode,Float64}(k,Base.Order.Reverse) #lower is better
        obj = new(k, q)
        obj
    end
end

"""
    optimize(p::GeneticProgram, grammar::Grammar, typ::Symbol, loss::Function; kwargs...)

Expression tree optimization using genetic programming with parameters p, grammar 'grammar', and start symbol typ, and loss function 'loss'.  Loss function has the form: los::Float64=loss(node::RuleNode, grammar::Grammar).
"""
function optimize(p::GeneticProgram, grammar::Grammar, typ::Symbol, loss::Function; kwargs...) 
    genetic_program(p, grammar, typ, loss; kwargs...)
end

"""
    genetic_program(p::GeneticProgram, grammar::Grammar, typ::Symbol, loss::Function)

Strongly-typed genetic programming with parameters p, grammar 'grammar', start symbol typ, and loss function 'loss'. Loss funciton has the form: los::Float64=loss(node::RuleNode, grammar::Grammar). 
    
See: Montana, "Strongly-typed genetic programming", Evolutionary Computation, Vol 3, Issue 2, 1995.
Koza, "Genetic programming: on the programming of computers by means of natural selection", MIT Press, 1992 

Three operators are implemented: reproduction, crossover, and mutation.
"""
function genetic_program(p::GeneticProgram, grammar::Grammar, typ::Symbol, loss::Function; 
    verbose::Bool=false, save::Bool=true, resume=true)
    dmap = mindepth_map(grammar)
    if resume && isfile("pop.bson")
        pop0 = Vector{RuleNode}(BSON.load("pop.bson")[:pop])
    else
        pop0 = initialize(p.init_method, p.pop_size, grammar, typ, dmap, p.max_depth)
    end
    pop1 = Vector{RuleNode}(undef,p.pop_size)
    losses0 = Vector{Union{Float64,Missing}}(missing,p.pop_size)
    losses1 = Vector{Union{Float64,Missing}}(missing,p.pop_size)

    best_tree, best_loss = evaluate!(p, loss, grammar, pop0, losses0, pop0[1], Inf; verbose=verbose)
    for iter = 1:p.iterations
        @master if verbose 
            best_expr = get_executable(best_tree, grammar)
            best_val_loss = loss(best_tree, grammar, train = false)
            print("iterations: $iter of $(p.iterations), best_loss: $best_loss, ")
            println("best_val_loss: $best_val_loss, best_expr: $best_expr")
        end
        fill!(losses1, missing)
        i = 0
        while i < p.pop_size
            op = sample(OPERATORS, p.p_operators)
            if op == :reproduction
                ind1,j = select(p.select_method, pop0, losses0)
                pop1[i+=1] = ind1
                losses1[i] = losses0[j]
            elseif op == :crossover
                ind1,_ = select(p.select_method, pop0, losses0)
                ind2,_ = select(p.select_method, pop0, losses0)
                child = crossover(ind1, ind2, grammar, p.max_depth)
                pop1[i+=1] = child
            elseif op == :subtree_mutation
                ind1,_ = select(p.select_method, pop0, losses0)
                child1 = subtree_mutation(ind1, grammar, dmap, p.max_depth)
                pop1[i+=1] = child1
            elseif op == :hoist_mutation
                ind1,_ = select(p.select_method, pop0, losses0)
                child1 = hoist_mutation(ind1, grammar, dmap, p.max_depth)
                pop1[i+=1] = child1                
            elseif op == :point_mutation
                ind1,_ = select(p.select_method, pop0, losses0)
                child1 = point_mutation(ind1, grammar, p.point_replace)
                pop1[i+=1] = child1
            end
        end
        pop0, pop1 = pop1, pop0
        losses0, losses1 = losses1, losses0
        best_tree, best_loss = evaluate!(p, loss, grammar, pop0, losses0, best_tree, best_loss; verbose=verbose)
        @master save && BSON.bson("pop.bson", pop = pop0, expr = string.(get_executable.(pop0, Ref(grammar))))
    end
    alg_result = Dict{Symbol,Any}()
    _add_result!(alg_result, p.track_method)
    ExprOptResult(best_tree, best_loss, get_executable(best_tree, grammar), alg_result)
end

"""
    _add_result!(d::Dict{Symbol,Any}, t::NoTracking)

Add tracking results to alg_result.  No op for NoTracking.
"""
_add_result!(d::Dict{Symbol,Any}, t::NoTracking) = nothing
"""
    _add_result!(d::Dict{Symbol,Any}, t::TopKTracking)

Add tracking results to alg_result. 
"""
function _add_result!(d::Dict{Symbol,Any}, t::TopKTracking)
    d[:top_k] = collect(t.q)
    d
end

"""
    initialize(::RandomInit, pop_size::Int, grammar::Grammar, typ::Symbol, dmap::AbstractVector{Int}, 
max_depth::Int)

Random population initialization.
"""
function initialize(::RandomInit, pop_size::Int, grammar::Grammar, typ::Symbol, 
    dmap::AbstractVector{Int}, max_depth::Int)
    [randtree(grammar, typ, dmap, rand(1:max_depth)) for i = 1:pop_size]
end

"""
    select(p::TournamentSelection, pop::Vector{RuleNode}, losses::Vector{Union{Float64,Missing}})

Tournament selection.
"""
function select(p::TournamentSelection, pop::Vector{RuleNode}, 
    losses::Vector{Union{Float64,Missing}})
    ids = StatsBase.sample(1:length(pop), p.k; replace=false, ordered=true) 
    i = ids[1] #assumes pop is sorted
    pop[i], i
end

"""
    select(p::TruncationSelection, pop::Vector{RuleNode}, losses::Vector{Union{Float64,Missing}})

Truncation selection.
"""
function select(p::TruncationSelection, pop::Vector{RuleNode}, 
    losses::Vector{Union{Float64,Missing}})
    i = rand(1:p.k)  #assumes pop is sorted
    pop[i], i
end

"""
    evaluate!(p::GeneticProgram, loss::Function, grammar::Grammar, pop::Vector{RuleNode}, losses::Vector{Union{Float64,Missing}}, 
        best_tree::RuleNode, best_loss::Float64)

Evaluate the loss function for population and sort.  Update the globally best tree, if needed.
"""
function evaluate!(p::GeneticProgram, loss::Function, grammar::Grammar, pop::Vector{RuleNode}, 
    losses::Vector{Union{Float64,Missing}}, best_tree::RuleNode, best_loss::Float64; verbose=false)

    # Pre-compute indics that are missing to make
    # the loop multi-threading friendly.
    idcs_missing = filter(i -> ismissing(losses[i]), eachindex(pop))
    prog = Progress(length(idcs_missing), desc="Evaluating: ")
    fmap = GCMAES.worldsize() > 1 ? GCMAES.pmap : map
    losses[idcs_missing] .= fmap(idcs_missing) do i
        l = loss(pop[i], grammar)
        pen = p.parsimony_coefficient * length(pop[i])
        if GCMAES.myrank() == 0
            ex = get_executable(pop[i], grammar)
            next!(prog, showvalues = [(:loss, l), (:penalty, pen), (:expr, ex)])
        end
        return l + pen
    end

    perm = sortperm(losses)
    pop[:], losses[:] = pop[perm], losses[perm]
    if losses[1] < best_loss
        best_tree, best_loss = pop[1], losses[1]
    end
    _update_tracker!(p.track_method, pop, losses)
    (best_tree, best_loss)
end

"""
    _update_tracker!(t::NoTracking, pop::Vector{RuleNode}, losses::Vector{Union{Float64,Missing}}) 

Update the tracker.  No op for NoTracking.
"""
function _update_tracker!(t::NoTracking, pop::Vector{RuleNode}, 
    losses::Vector{Union{Float64,Missing}}) 
    nothing
end
"""
    _update_tracker!(t::TopKTracking, pop::Vector{RuleNode}, losses::Vector{Union{Float64,Missing}})

Update the tracker.  Track top k expressions. 
"""
function _update_tracker!(t::TopKTracking, pop::Vector{RuleNode}, 
    losses::Vector{Union{Float64,Missing}})
    n = 0
    for i = 1:length(pop)
        r = enqueue!(t.q, pop[i], losses[i])
        r >= 0 && (n += 1) #no clash, increment counter
        n >= t.k && break 
    end
end

"""
    crossover(a::RuleNode, b::RuleNode, grammar::Grammar)

Crossover genetic operator.  Pick a random node from 'a', then pick a random node from 'b' that has the same type, then replace the subtree 
"""
function crossover(a::RuleNode, b::RuleNode, grammar::Grammar, max_depth::Int)
    child = deepcopy(a)
    crosspoint = sample(b)
    typ = return_type(grammar, crosspoint.ind)
    d_subtree = depth(crosspoint)
    d_max = max_depth + 1 - d_subtree 
    if d_max > 0 && contains_returntype(child, grammar, typ, d_max)
        loc = sample(NodeLoc, child, typ, grammar, d_max)
        insert!(child, loc, deepcopy(crosspoint))
    end
    child 
end

"""
subtree_mutation(a::RuleNode, grammar::Grammar, dmap::AbstractVector{Int}, max_depth::Int=5)

Subtree Mutation genetic operator.  Pick a random node from 'a', then replace the subtree with a random one.
"""
function subtree_mutation(a::RuleNode, grammar::Grammar, dmap::AbstractVector{Int}, max_depth::Int)
    child = deepcopy(a)
    loc = sample(NodeLoc, child)
    mutatepoint = get(child, loc) 
    typ = return_type(grammar, mutatepoint.ind)
    d_node = node_depth(child, mutatepoint)
    d_max = max_depth + 1 - d_node
    if d_max > 0
        subtree = randtree(grammar, typ, dmap, rand(1:d_max))
        insert!(child, loc, subtree)
    end
    child
end

"""
hoist_mutation(a::RuleNode, grammar::Grammar, dmap::AbstractVector{Int}, max_depth::Int=5)

Hoist Mutation genetic operator. Take the winner of a tournament and select a random subtree from it.
A random subtree of that subtree is then selected and this is hoisted into the original subtrees location 
to form an offspring in the next generation. 
"""
function hoist_mutation(a::RuleNode, grammar::Grammar, dmap::AbstractVector{Int}, max_depth::Int)
    child = deepcopy(a)
    for i in 1:3
        loc = sample(NodeLoc, child)
        subtree = get(child, loc)
        for j in 1:3
            subloc = sample(NodeLoc, subtree)
            subsubtree = get(subtree, subloc)
            if return_type(grammar, subsubtree) == return_type(grammar, subtree)
                insert!(child, loc, subsubtree)
                return child
            end
        end
    end
    child
end

"""
point_mutation(a::RuleNode, grammar::Grammar, dmap::AbstractVector{Int}, max_depth::Int=5)

Point Mutation genetic operator. Point Mutation takes the winner of a tournament and selects 
random nodes from it to be replaced. Terminals are replaced by other terminals and functions 
are replaced by other functions that require the same number of arguments as the original node. 
The resulting tree forms an offspring in the next generation.
"""
point_mutation(a::RuleNode, grammar::Grammar, point_replace::Float64) = 
    point_mutation!(deepcopy(a), grammar, point_mutation)
function point_mutation!(a, grammar, point_replace)
    if rand() < point_replace
        a._val = iseval(grammar, a) ? Core.eval(a, grammar) : a._val
        a.ind = filter(eachindex(grammar.rules)) do r
            return_type(grammar, r) == return_type(grammar, a) &&
            child_types(garmmar, r) == child_types(grammar, a)
        end |> rand
    end
    for c in a.children
        point_mutation!(c, grammar, point_mutation)
    end
    return a
end

function randtree(grammar, typ, dmap, d_max)
    if rand() < 0.5
        rand(RuleNode, grammar, typ, dmap, d_max)
    else
        rand_full(grammar, typ, dmap, d_max)
    end
end

function rand_full(grammar, typ, dmap, max_depth, bin=nothing)
    rules = grammar[typ]
    rules = filter(r->dmap[r] ≤ max_depth, rules)
    rules_nonterm = filter(r -> !isterminal(grammar, r), rules)
    if isempty(rules_nonterm)
        rule_index = StatsBase.sample(rules)
    else
        rule_index = StatsBase.sample(rules_nonterm)
    end

    rulenode = iseval(grammar, rule_index) ?
        RuleNode(bin, rule_index, Core.eval(grammar, rule_index)) :
        RuleNode(bin, rule_index)

    if !grammar.isterminal[rule_index]
        for ch in child_types(grammar, rule_index)
            push!(rulenode.children, rand_full(grammar, ch, dmap, max_depth-1, bin))
        end
    end
    return rulenode
end

end #module
