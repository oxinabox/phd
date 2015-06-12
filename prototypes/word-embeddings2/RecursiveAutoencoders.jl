module RecursiveAutoencoders
using Compat


using WordEmbeddings
using Pipe

export  RAE,RAE_empty_like, get_word_index, eval_word_embedding,eval_word_embeddings, eval_merges, eval_scores, reconstruct, unfold_merges, unfold, eval_merge


type RAE{N, S<:AbstractString}<: Embedder
    L::Embeddings
    word_index::Dict{S,Int}
    indexed_words::Vector{S}
    
    W_e::Embeddings{N}
    b_e::Embedding{N}
    W_d::Embeddings{N}
    b_d::Embedding{N}
   
end


function RAE{N, S<:AbstractString}(L::Embeddings{N},word_index::Dict{S,Int}, indexed_words::Vector{S}, init_varience=0.01)
    emb_width = size(L,1)
    
    W_e = init_varience*randn(emb_width,emb_width*2) 
    b_e = init_varience*randn(emb_width) 
    W_d = init_varience*randn(emb_width*2,emb_width)
    b_d = init_varience*randn(emb_width*2)
    
    RAE(L,word_index, indexed_words, W_e, b_e, W_d, b_d)
end


function RAE_empty_like(rae::RAE)
    RAE([NaN]'', Dict{String,Int64}(), String[], rae.W_e, rae.b_e, rae.W_d, rae.b_d)
end

#-----Basic methods



function eval_merge{N}(rae::RAE, c_i::Embedding{N}, c_j::Embedding{N})
    c_ij = [c_i;c_j]
    ps=tanh(rae.W_e*c_ij.+rae.b_e)[:]
end


function eval_merges{N}(rae::RAE, c_ijs::Embeddings{N})
    ps=tanh(rae.W_e*c_ijs.+rae.b_e)
end

function eval_merges{N}(rae::RAE, c_is::Embeddings{N}, c_js::Embeddings{N})
    @assert size(c_is)==size(c_js)
    eval_merges(rae,[c_is;c_js])
end


function reconstruct{N}(rae::RAE, pp::Embedding{N})
    ĉ_ij = tanh(rae.W_d*pp+rae.b_d)
    ĉ_i::Embedding = ĉ_ij[1 : round(Int,end/2)]
    ĉ_j::Embedding = ĉ_ij[round(Int,end/2) + 1:end]
    ĉ_i, ĉ_j
end

function unfold_merges{N}(rae::RAE, pps::Embeddings{N})
    ĉ_ijs::Embeddings = tanh(rae.W_d*pps .+ rae.b_d)
end






end