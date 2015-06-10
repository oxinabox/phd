module RecursiveAutoencoders
using WordEmbeddings
using Pipe

export  RAE, get_word_index, eval_word_embedding,eval_word_embeddings, eval_merges, eval_scores, reconstruct, unfold_merges, unfold, eval_merge



type RAE<: Embedder
    L::Embeddings
    word_index::Dict{String,Int}
    indexed_words::Vector{String}
    
    W_e::Embeddings
    b_e::Embedding
    W_d::Embeddings
    b_d::Embedding
   
end


function RAE(L::Embeddings,word_index::Dict{String,Int}, indexed_words::Vector{String}, init_varience=0.01)
    emb_width = size(L,1)
    
    W_e = init_varience*randn(emb_width,emb_width*2) 
    b_e = init_varience*randn(emb_width) 
    W_d = init_varience*randn(emb_width*2,emb_width)
    b_d = init_varience*randn(emb_width*2)
    
    RAE(L,word_index, indexed_words, W_e, b_e, W_d, b_d)
end

#-----Basic methods


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
    ĉ_i::Embedding = ĉ_ij[1:end/2]
    ĉ_j::Embedding = ĉ_ij[end/2+1:end]
    ĉ_i, ĉ_j
end

function unfold_merges(rae::RAE, pps::Embeddings)
    ĉ_ijs::Embeddings = tanh(rae.W_d*pps .+ rae.b_d)
end






end