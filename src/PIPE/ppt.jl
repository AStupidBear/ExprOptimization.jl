using StatsBase

struct PPTParams
    w_terminal::Float64
    w_nonterm::Float64
end
PPTParams(w_terminal::Float64=0.6) = PPTParams(w_terminal, 1-w_terminal)

struct PPTNode
    ps::Dict{Symbol,Vector{Float64}}
    children::Vector{PPTNode}
end
function PPTNode(pp::PPTParams, grammar::Grammar)
    ps = Dict(typ=>normalize!([isterminal(grammar, i) ?
                               pp.w_terminal : pp.w_nonterm for i in grammar[typ]], 1)
             for typ in nonterminals(grammar))
    PPTNode(ps, PPTNode[])
end
function get_child(pp::PPTParams, ppt::PPTNode, grammar::Grammar, i::Int)
    if i > length(ppt.children)
        push!(ppt.children, PPTNode(pp, grammar))
    end
    ppt.children[i]
end

function Base.rand(pp::PPTParams, ppt::PPTNode, grammar::Grammar, typ::Symbol)
    rules = grammar[typ]
    rule_index = sample(rules, Weights(ppt.ps[typ]))
    ctypes = child_types(grammar, rule_index)
    node = iseval(grammar, rule_index) ? 
        RuleNode(rule_index, eval(grammar, rule_index), Vector{RuleNode}(length(ctypes))) :
        RuleNode(rule_index, Vector{RuleNode}(length(ctypes)))

    for (i,typ) in enumerate(ctypes)
        node.children[i] = rand(pp, get_child(pp, ppt, grammar, i), grammar, typ)
    end
    node
end
function probability(pp::PPTParams, ppt::PPTNode, grammar::Grammar, expr::RuleNode)
    typ = return_type(grammar, expr)
    i = findfirst(grammar[typ], expr.ind)
    retval = ppt.ps[typ][i]
    for (i,c) in enumerate(expr.children)
        retval *= probability(pp, get_child(pp, ppt, grammar, i), grammar, c)
    end
    retval
end

function prune!(ppt::PPTNode, grammar::Grammar, p_threshold::Float64)
    kmax, pmax = :None, 0.0
    for (k, p) in ppt.ps
        pmax′ = maximum(p)
        if pmax′ > pmax
            kmax, pmax = k, pmax′
        end
    end
    if pmax > p_threshold
        i = indmax(ppt.ps[kmax])
        if isterminal(grammar, i)
            clear!(ppt.children)
        else
            max_arity_for_rule = maximum(nchildren(grammar, r) for
                                         r in grammar[kmax])
            while length(ppt.children) > max_arity_for_rule
                pop!(ppt.children)
            end
        end
    end
    return ppt
end
