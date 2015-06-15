module UnfoldingRAE
using Compat


using RecursiveAutoencoders
using WordEmbeddings
using Base.Collections

export UnfoldLeaf, FoldData, UnfoldData, fold, unfold, UBPTS, loss,loss_and_loss_grad

abstract Side

immutable Left<:Side
end

immutable Right<:Side
end

immutable NoSide<:Side
end

immutable FoldData
    p_out::Embedding
    left::Union(FoldData,Embedding)
    right::Union(FoldData,Embedding)
end

immutable UnfoldData{T<:Side}
    p_in::Embedding
    parent::Union(FoldData,UnfoldData)
    ĉ_i::Embedding
    ĉ_j::Embedding
    depth::Int64
end

immutable UnfoldLeaf{T<:Side} 
    ĉ::Embedding
    parent::UnfoldData
    c::Embedding
    depth::Int64
end

function get_side{T}(::Union(UnfoldLeaf{T}, UnfoldData{T}))
    T()
end


function emb(data::FoldData)
    data.p_out
end
function emb(data::Embedding)
    data
end


#------------------____FOLDING/UNFOLDING________---------------

function fold(rae::RAE, tree::@compat Tuple{Any,Any})
    function eval_child(child::String)
        c=eval_word_embedding(rae,child,false)
        c::Embedding
    end
    function eval_child(c::Embedding)
        c::Embedding
    end
    function eval_child(tree::@compat Tuple{Any,Any})
        fold(rae,tree)
    end
    
   
    const left = eval_child(tree[1])
    const right = eval_child(tree[2])
    const p=eval_merge(rae, emb(left), emb(right))
    FoldData(p, left, right)   
end

function unfold{T}(rae::RAE, c::Embedding, ĉ::Embedding, parent, ::Type{T}, depth)
    UnfoldLeaf{T}(ĉ, parent, c, depth)
end


function unfold{T}(rae::RAE, act::FoldData, p_in::Embedding, parent, ::Type{T}, depth::Int)
    #Side is a ignored argument. This could be replaced with a generated function
    const ĉ_i, ĉ_j = reconstruct(rae,p_in)
    const data = UnfoldData{T}(p_in, parent, ĉ_i, ĉ_j,depth)
    
    const left = unfold(rae, act.left, ĉ_i, data, Left, depth+1)
    const right= unfold(rae, act.right, ĉ_j, data, Right, depth+1)
    [left; right]
end

function unfold(rae::RAE, act::FoldData)
    #Handle the top case
    unfold(rae, act,act.p_out,act, NoSide,0)
end


#---------------______GRADIENT___________--------------------

function δ(a::Embedding, δ_above::NumericVector, W::NumericMatrix)
    #a is the ouput of this layer: a=tanh(z) where z is the input from layer below
    #W is matrix to move to above layer, from this one
    const dz = 1-a.^2 #Derivitive of a=tanh(z)
    (W'*δ_above).*dz
end

function δ(ĉ_ij::Embedding,c_ij::Embedding) 
    #Output Layer
    const M = length(c_ij)# ==length(ĉ_ij)
    const dz = 1-ĉ_ij.^2
    const δ_above = -(c_ij-ĉ_ij)
    δ_above.*dz
    #δ(ĉ_ij,δ_above, eye(M))     
end


function sidepad(d::NumericVector, ::Left)
    const padding=zeros(d)
    [d; padding]
end
function sidepad(d::NumericVector, ::Right)
    const padding=zeros(d)
    [padding; d]
end

function sidepad(d::NumericVector, ::NoSide)
    d
end


function UBPTS(rae::RAE, nodes::Vector{UnfoldLeaf} )
    parent_deltas = Dict{UnfoldData, NumericVector}()
    function add!(parent_node, delta)
        if haskey(parent_deltas, parent_node)
            parent_deltas[parent_node]+=delta
        else
            parent_deltas[parent_node]=delta
        end
    end
    
    @inbounds for leaf in nodes
        const δ_node::NumericVector = δ(leaf.ĉ,leaf.c)
        const δ_padded = sidepad(δ_node, get_side(leaf))
        add!(leaf.parent, δ_padded)
    end
        
    UBPTS(rae,parent_deltas)
end

function UBPTS(rae::RAE, parent_deltas::Dict{UnfoldData,NumericVector})
    foldnode = nothing
    δ_above_fold = 0
    
    #pending_nodes = PriorityQueue(UnfoldData, Int64,Base.Order.Reverse) #0.4 style
    pending_nodes = PriorityQueue{UnfoldData, Int64}(Base.Order.Reverse) #0.3 style
    enqueue!(node::UnfoldData) = pending_nodes[node] = node.depth #Priority of node.depth (syntax on julia Priority queues is weird)
    map(enqueue!, keys(parent_deltas)) #Add all that were passed, as none have been processed
    
    function pend!(parent_node::UnfoldData, δ_node::NumericVector)
        if !haskey(parent_deltas,parent_node)
            enqueue!(parent_node) #then also hasn't been enque
            parent_deltas[parent_node]=δ_node
        else
            parent_deltas[parent_node]+=δ_node
        end
    end
        
    function pend!(node::FoldData, δ_node::NumericVector)
        foldnode = node
        δ_above_fold+=δ_node
    end

    ΔW_d=0 #will broadcast
    Δb_d=0 
    while !isempty(pending_nodes)
        node = dequeue!(pending_nodes)
        δ_above::NumericVector =  parent_deltas[node]
        #Note: node.p_in= suitable half of node.parent.ĉ_i or node.parent.ĉ_j
        #      The line below takes a lot of thinking to be sure it is right
        δ_node::NumericVector = δ(node.p_in, δ_above, rae.W_d)
        δ_padded = sidepad(δ_node, get_side(node))
        
        ΔW_d += δ_above*node.p_in'
        Δb_d += δ_above
        pend!(node.parent,δ_padded)
    end

    (δ_above_fold, ΔW_d, Δb_d)
end

function UBPTS(rae::RAE, node::FoldData, δ_above::NumericVector)
    const c_i=emb(node.left)
    const c_j=emb(node.right)
    const a= [c_i; c_j]
    
    const δ_node::NumericVector =  δ(a, δ_above, rae.W_e)
    const δ_left::NumericVector = δ_node[1:end÷2]
    const δ_right::NumericVector = δ_node[1+ end÷2 : end]
    
    const ΔW_e=δ_above*a'
    const Δb_e=δ_above   
    
    const ΔW_e_left, Δb_e_left = UBPTS(rae, node.left, δ_left)
    const ΔW_e_right, Δb_e_right = UBPTS(rae, node.right, δ_right)
    (ΔW_e+ΔW_e_left+ΔW_e_right, Δb_e+Δb_e_left+Δb_e_right)
end

function UBPTS(rae::RAE, node::Embedding, δ_above::NumericVector)
    0,0,0 # Nothing to learn here (at least until we start learning rae.L)
end

#----------------__Loss Functions___---------
function loss(unfold_leaves::Vector{UnfoldLeaf})
    map(unfold_leaves) do leaf
        0.5*(leaf.c-leaf.ĉ).^2 |> sum
        end |> sum 
end

function loss(rae::RAE, tree::@compat Tuple{Any,Any})
    const fold_tree = fold(rae, tree)
    const unfold_leaves = unfold(rae, fold_tree)
    loss(unfold_leaves)
end


function loss_and_loss_grad(rae::RAE, tree::@compat Tuple{Any,Any})
    const fold_tree = fold(rae, tree)
    const unfold_leaves = unfold(rae, fold_tree)
    const err=loss(unfold_leaves)

    const δd,∇W_d, ∇b_d = UBPTS(rae, unfold_leaves)
    const ∇W_e,∇b_e = UBPTS(rae, fold_tree, δd)

    const Δs = (∇W_e, ∇b_e, ∇W_d, ∇b_d)
    (Δs, err)
end




end