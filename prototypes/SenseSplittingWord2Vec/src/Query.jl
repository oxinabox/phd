module Query
using Base.Collections
using WordEmbeddings
using Distances
using SoftmaxClassifier
export find_nearest_words, prob_of_context


########### nearest_words, and analogy math

function find_nearest_words(embed::GenWordEmbedding, equation::AbstractString; nwords=5)
	tokens = replace(replace(equation, "+", " + "), "-", " - ")
    positive_words = AbstractString[]
    negative_words = AbstractString[]
    wordlist = positive_words

    for tok in split(tokens)
        tok = strip(tok)
        isempty(tok) && continue
        if tok == "+"
            wordlist = positive_words #change to putting in the positive list
        elseif tok == "-"
            wordlist = negative_words #change to putting in the negitive list
        else
            push!(wordlist, tok)      #It must be a word, put it in currnet list
        end
    end
	
	find_nearest_words(embed, positive_words, negative_words; nwords=nwords)
end

function find_nearest_words(embed::GenWordEmbedding, positive_words::Vector, negative_words::Vector; nwords=5)
    pq = PriorityQueue(Base.Order.Reverse)
    wv = sum([embed.embedding[w] for w∈positive_words])
	wv .-= length(negative_words)>0 ? sum([embed.embedding[w] for w∈negative_words]) : 0.0

    for (w, embed_w) in embed.embedding
        if (w in positive_words) || (w in negative_words)
            continue
        end
        dist = cosine_dist(wv, embed_w)
        enqueue!(pq, w, dist)
        if length(pq) > nwords
            dequeue!(pq)
        end
    end
    sort(collect(pq), by = t -> t[2])
end

#################### Probability of the context
    
function prob_of_context{S<:AbstractString}(embed::WordEmbedding, context::AbstractVector{S}, middle_word::S)
    input = embed.embedding[middle_word]
    prob_of_context(embed, context, input)
end

function prob_of_context{S<:AbstractString}(embed::GenWordEmbedding, context::AbstractVector{S}, input::Vector{Float32})
    total_logprob=0.0 #Work in logprob to avoid underflow, and get more stability
    for target_word in context
        # discard words not presenting in the classification tree
        haskey(embed.codebook, target_word) || continue
        node = embed.classification_tree      
        
        word_logprob = 0.0
        for code in embed.codebook[target_word]  
            word_logprob+=log(predict(node.data, input)[code])
            node = node.children[code]
        end
        total_logprob+=word_logprob
    end
    exp(total_logprob) #Going back out of the log domain is not required for external logic, but it is nice for clarity
end


end #Module
