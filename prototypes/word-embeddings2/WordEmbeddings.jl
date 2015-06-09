module WordEmbeddings

using Pipe

export Words, Embedding, Embeddings, load_embeddings, cosine_dist, neighbour_dists,show_best, show_bests, WE, Embedder, get_word_index, eval_word_embedding, eval_word_embeddings, load_word2vec_embeddings, has_word

typealias Words Union(AbstractArray{ASCIIString,1},AbstractArray{String,1})
typealias Embedding Union(Vector{Float64}, Vector{Float32}) 
typealias Embeddings Union(Matrix{Float64},Matrix{Float32})

const UNKNOWN_WORD = "*UNKNOWN*"

#Loads Turins embeddings
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

#Loads googles word2vec_embeddings
function load_word2vec_embeddings(embedding_file, max_stored_vocab_size = 1000000)
    #Note: I know there actually <10^6 words in the vocab, when phrases are exlusded, so lock the vocab size to this to save 70%RAM
    #Words are loosely organised by commonness,  AFAICT
    fh = open(embedding_file,"r")
    vocab_size, vector_size = @pipe readline(fh)|> split |> map(int, _)
    max_stored_vocab_size = min(max_stored_vocab_size, vocab_size) #if using a small vocab then there is a chance you might be willing ot store more words than it has
    
    
    indexed_words = Array(String,max_stored_vocab_size)
    word_indexes = Dict{String,Int64}()
    LL = Array(Float32,(vector_size, max_stored_vocab_size))


    
    #Add a Zero vector for the unknown words
    LL[:,1]*=0
    indexed_words[1]=UNKNOWN_WORD
    word_indexes[UNKNOWN_WORD]=1

    #Add all from data
    index = 2
    for _ in 1:vocab_size
        word = readuntil(fh,' ') |> strip #Technically this is 'ISO-8859-1' may have to deal with encoding issues
        vector = read(fh, Float32,vector_size ) 

        if !contains(word, "_") #If it isn't a phrase
            LL[:,index]=vector./norm(vector)
            indexed_words[index] = word
            word_indexes[word] = index
            
            index+=1
            if index>max_stored_vocab_size
                break
            end
        end
        
    end
    LL = LL[:,1:index-1] #throw away unused columns
    indexed_words = indexed_words[1:index-1] #throw away unused columns
    LL,word_indexes, indexed_words
end

#----

abstract Embedder
immutable WE<:Embedder
    L::Matrix{Float64}
    word_index::Dict{String,Int}
    indexed_words::Vector{String}
end




function has_word(we::Embedder, input::String)
    haskey(we.word_index, input) || 
    haskey(we.word_index, lowercase(input)) ||
    haskey(we.word_index, uppercase(input[1:1])*input[2:end])
end

function get_word_index(we::Embedder, input::String, show_warn=true)
    if haskey(we.word_index, input) #Direct
        ii = we.word_index[input]
    elseif haskey(we.word_index, lowercase(input)) #remove capitals
        ii = we.word_index[lowercase(input)]
    elseif haskey(we.word_index, uppercase(input[1:1])*input[2:end]) # add capital at start (eg if a name)
        ii = we.word_index[uppercase(input[1:1])*input[2:end]]
    else
        if show_warn
            warn("$input not found. Defaulting.")
        end
        ii = we.word_index[UNKNOWN_WORD]
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

function neighbour_sims(cc::Embedding, globe::Embeddings, similarity=cosine_sim)
    [similarity(cc, globe[:,ii]) for ii in 1:size(globe,2)]
end


function show_best(embedder,ĉ::Embedding, nbest=20, similarity=cosine_sim )
    candidates=neighbour_sims(ĉ,embedder.L, similarity)   
    best_cands = [ (findfirst(candidates,score), score)
                    for score in select(candidates,1:nbest, rev=true)[1:nbest]]
    vcat([[embedder.indexed_words[ii] round(score,2)] for (ii,score) in best_cands]...)
end

function show_bests(embedder,ĉs::Embeddings, nbest=20, similarity=cosine_sim)
    hcat([show_best(embedder,ĉs[:,ii],nbest, similarity) for ii in 1:size(ĉs,2)]...)
end


end