module WordEmbeddings

using Pipe

export Words, Embedding, Embeddings, load_embeddings, cosine_dist, neighbour_dists,show_best, show_bests, WE, Embedder, get_word_index, eval_word_embedding, eval_word_embeddings

typealias Words Union(AbstractArray{ASCIIString,1},AbstractArray{String,1})
typealias Embedding Vector{Float64}
typealias Embeddings Matrix{Float64}

#returns `LL` embedding matrix and onehot lookup `ee`,
#Such that `LL*ee["example"]` returns the word embedding for "example"
function load_embeddings(embedding_file)
    embeddingsDict = Dict{String,Embedding}()
    #sizehint!(embeddings, 268810)
    for line in eachline(open(embedding_file))
        fields = line |> split
        word = fields[1]
        vec = map(fs -> parse(Float64,fs), fields[2:end])
        embeddingsDict[word] = vec
    end
    
    LL = hcat(collect(values(embeddingsDict))...)
    word_indexes = [word=>ii for (ii,word) in enumerate(keys(embeddingsDict))]  #Dict mapping Word to Index
    indexed_words = embeddingsDict |> keys |> collect # Vector mapping index to string
   
    LL,word_indexes, indexed_words
end

#----

abstract Embedder
type WE<:Embedder
    L::Matrix{Float64}
    word_index::Dict{String,Int}
    indexed_words::Vector{String}
end

function get_word_index(we::Embedder, input::String, show_warn=true)
    if haskey(we.word_index, input)
        ii = we.word_index[input]
    elseif haskey(we.word_index, lowercase(input))
        ii = we.word_index[lowercase(input)]
    else
        ii = we.word_index["*UNKNOWN*"]
        if show_warn
            warn("$input not found. Defaulting.")
        end
    end
    ii
end

function eval_word_embedding(we::Embedder, input::String, show_warn=true)
    k=get_word_index(we, input, show_warn)
    we.L[:,k]
end

function eval_word_embeddings(we::Embedder, input::Words, show_warn=true)
    @pipe input|> map(x->eval_word_embedding(we,x, show_warn), _) |> hcat(_...)
end


#----------- Eval nearest tools 

function cosine_sim(a,b) #This is actually the definition of cosign similarity
    (a⋅b)/(norm(a)*norm(b))
end

function neighbour_sims(cc::Vector{Float64}, globe::Matrix{Float64})
    [cosine_sim(cc, globe[:,ii]) for ii in 1:size(globe,2)]
end


function show_best(embedder,ĉ::Embedding, nbest=20)
    candidates=neighbour_sims(ĉ,embedder.L)   
    best_cands = [ (findfirst(candidates,score), score)
                    for score in select(candidates,1:nbest, rev=true)[1:nbest]]
    vcat([[embedder.indexed_words[ii] round(score,2)] for (ii,score) in best_cands]...)
end

function show_bests(embedder,ĉs::Embeddings, nbest=20)
    hcat([show_best(embedder,ĉs[:,ii],nbest) for ii in 1:size(ĉs,2)]...)
end


end