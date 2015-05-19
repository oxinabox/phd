module WordEmbeddings


export Embedding, Embeddings, load_embeddings

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
    embeddingsDict
    
    LL = hcat(collect(values(embeddingsDict))...)
    
    word_indexes = [word=>ii for (ii,word) in enumerate(keys(embeddingsDict))]  #Dict mapping Word to Index
    
    
    indexed_words = embeddingsDict |> keys |> collect # Vector mapping index to string
    
    
    LL,word_indexes, indexed_words
end




end