__precompile__()

module ExprOptimization

using ExprRules

export 
        @ruleset,
        RuleSet,
        RuleNode,
        get_executable,

        loss,
        optimize,
        ExprOptResults,

        MonteCarloParams,

        GeneticProgramParams,
        RandomInit,
        TournamentSelection

abstract type ExprOptParams end

struct ExprOptResults
    tree::RuleNode
    loss::Float64
    expr::Any
    alg_results::Any
end

#functions implemented by user
function loss end       #loss function, loss(tree::RuleNode)

"""
optimize(p::ExprOptParams, ruleset::RuleSet, typ::Symbol)

Main entry for expression optimization.  Use concrete ExprOptParams to specify optimization algorithm.
"""
function optimize end   #implemented by algorithms

include("MonteCarlo/monte_carlo.jl")
using .MonteCarlo

include("GeneticProgram/genetic_program.jl")
using .GeneticProgram

end # module
