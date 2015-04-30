#returns `LL` embedding matrix and onehot lookup `ee`,
#Such that `LL*ee["example"]` returns the word embedding for "example"
function load_embeddings(embedding_file)
    embeddingsDict = Dict{String,Vector{Float64}}()
    #sizehint!(embeddings, 268810)
    for line in eachline(open(embedding_file))
        fields = line |> split
        word = fields[1]
        vec = map(parsefloat, fields[2:end])
        embeddingsDict[word] = vec
    end
    embeddingsDict
    
    LL = hcat(collect(values(embeddingsDict))...)
    
    word_indexes = [word=>ii for (ii,word) in enumerate(keys(embeddingsDict))]  #Dict mapping Word to Index
    
    
    
    #word_index_vectors::Dict{String,BitVector} = [key=>setindex!(BitArray(length(embeddingsDict)), true, ii) 
    #            for (ii,key) in enumerate(keys(embeddingsDict))]
    
    
    indexed_words = embeddingsDict |> keys |> collect # Vector mapping index to string
    
    
    LL,word_indexes, indexed_words
end