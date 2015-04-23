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
    ee = [key=>setindex!(BitArray(length(embeddingsDict)), true, ii) 
                for (ii,key) in enumerate(keys(embeddingsDict))]
    indexed_word = embeddingsDict |> keys |> collect
    
    LL,ee, indexed_word
end