module RecursiveAutoencoders
using WordEmbeddings
using Pipe

export Words, RAE, get_word_index, eval_word_embedding,eval_word_embeddings, eval_merges, eval_scores, reconstruct, unfold_merges, ActData, eval_to_tree, BPTS, eval_scores_gradient, unfold, eval_merge


typealias Words Union(AbstractArray{ASCIIString,1},AbstractArray{String,1})
type RAE
    L::Matrix{Float64}
    word_index::Dict{String,Int}
    indexed_words::Vector{String}
    
    W_e::Matrix{Float64}
    b_e::Vector{Float64}
    W_d::Matrix{Float64}
    b_d::Vector{Float64}
   
end


function RAE(L::Matrix{Float64},word_index::Dict{String,Int}, indexed_words::Vector{String})
    emb_width = size(L,1)
    
    W_e =0.01*randn(emb_width,emb_width*2) 
    b_e = 0.01*randn(emb_width) 
    #W_d = 0.01*randn(emb_width*2,emb_width)
    W_d = pinv(W_e) #Cheat (Actually why can't I always do this to initialize?);
    b_d = 0.01*randn(emb_width*2)
    
    RAE(L,word_index, indexed_words, W_e, b_e, W_d, b_d)
end

#-----Basic methods

function get_word_index(rae::RAE, input::String, show_warn=true)
    if haskey(rae.word_index, input)
        ii = rae.word_index[input]
    elseif haskey(rae.word_index, lowercase(input))
        ii = rae.word_index[lowercase(input)]
    else
        ii = rae.word_index["*UNKNOWN*"]
        if show_warn
            println("$input not found. Defaulting.")
        end
    end
    ii
end


function eval_word_embedding(rae::RAE, input::String, show_warn=true)
    k=get_word_index(rae, input, show_warn)
    rae.L[:,k]
end

function eval_word_embeddings(rae::RAE, inputs::Words, show_warn=false)
    ks = @pipe inputs |> map(ii -> get_word_index(rae,ii, show_warn), _)
    rae.L[:,ks]
end


function eval_word_embeddings(rae::RAE, tree::(Any,Any))
    function eval_child(child::String)
        eval_word_embedding(rae,child,false)
    end
    function eval_child(child::Any)
        eval_word_embeddings(rae,child)
    end
    c_i = eval_child(tree[1])
    c_j = eval_child(tree[2])
    [c_i c_j]
end

function eval_merge(rae::RAE, c_i::Embedding, c_j::Embedding)
    c_ij = [c_i;c_j]
    ps=tanh(rae.W_e*c_ij.+rae.b_e)[:]
end


function eval_merges(rae::RAE, c_ijs::Embeddings)
    ps=tanh(rae.W_e*c_ijs.+rae.b_e)
end

function eval_merges(rae::RAE, c_is::Embeddings, c_js::Embeddings)
    @assert size(c_is)==size(c_js)
    eval_merges(rae,[c_is;c_js])
end

function eval_scores(rae::RAE, c_is::Embeddings, c_js::Embeddings,
                      pps=eval_merges(rae, c_is, c_js)::Embeddings,
                      ĉ_ijs = unfold_merges(rae,pps)::Embeddings)
     c_ijs = [c_is;c_js]
     
     1/2*sum((c_ijs-ĉ_ijs).^2,1)
end

function reconstruct(rae::RAE, pp::Embedding)
    ĉ_ij = tanh(rae.W_d*pp+rae.b_d)
    ĉ_i = ĉ_ij[1:end/2]
    ĉ_j = ĉ_ij[end/2+1:end]
    ĉ_i, ĉ_j
end

function unfold_merges(rae::RAE, pps::Embeddings)
    ĉ_ijs = tanh(rae.W_d*pps .+ rae.b_d)
end

#----------- Eval nearest tools 

function cosine_dist(a,b)
    (a⋅b)/(norm(a)*norm(b))
end

function neighbour_dists(cc::Vector{Float64}, globe::Matrix{Float64})
    [cosine_dist(cc, globe[:,ii]) for ii in 1:size(globe,2)]
end


function show_best(rae::RAE,ĉ::Embedding, nbest=20)
    candidates=neighbour_dists(ĉ,rae.L)   
    best_cands = [ (findfirst(candidates,score), score)
                    for score in select(candidates,1:nbest, rev=true)[1:nbest]]
    vcat([[rae.indexed_words[ii] round(score,2)] for (ii,score) in best_cands]...)
end

function show_bests(rae::RAE,ĉs::Embeddings, nbest=20)
    hcat([show_best(rae,ĉs[:,ii],nbest) for ii in 1:size(ĉs,2)]...)
end




end