module Query
using Base.Collections
using WordEmbeddings
using Distances
using SoftmaxClassifier
export find_nearest_words, logprob_of_context


########### nearest_words, and analogy math

function find_nearest_words(embed::WordEmbedding, equation::AbstractString; nwords=5)
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

function find_nearest_words(embed::WordEmbedding, positive_words::Vector, negative_words::Vector; nwords=5)
    wv = sum([embed.embedding[w] for w in positive_words])
	wv .-= length(negative_words)>0 ? sum([embed.embedding[w] for w in negative_words]) : 0.0

	find_nearest_embedding(embed.embedding, wv; nwords=nwords, banned=[positive_words;negative_words]) 
end


function find_nearest_words(embed::WordSenseEmbedding, word, sense_id; nwords=5)
	wv = embed.embedding[word][sense_id]
	flat_embeddings = flatten_embeddings(embed) 
	find_nearest_embedding(flat_embeddings, wv; nwords=nwords, banned=[(word,sense_id)]) 
end


function find_nearest_embedding(candidate_embeddings, target_embedding; nwords=5, banned=[])
    pq = PriorityQueue(Base.Order.Reverse)
    for (w, embed_w) in candidate_embeddings
		if w in banned
			continue
		end
        dist = cosine_dist(target_embedding, embed_w)
        enqueue!(pq, w, dist)
        if length(pq) > nwords
            dequeue!(pq)
        end
    end
    sort(collect(pq), by = t -> t[2])
end


#################### Probability of the context
    
function logprob_of_context{S<:AbstractString}(embed::WordEmbedding, context::AbstractVector{S}, middle_word::S; kwargs...)
    input = embed.embedding[middle_word]
    logprob_of_context(embed, context, input; kwargs...)
end


function logprob_of_context{S<:AbstractString}(embed::GenWordEmbedding, context::AbstractVector{S}, input::Vector{Float32}; skip_oov=false)
    total_prob=0.0f0
    for target_word in context
		skip_oov && !haskey(embed.codebook, target_word) && continue
        node = embed.classification_tree      
        
        word_prob = 0.0f0
        for code in embed.codebook[target_word]  
            word_prob+= log(predict(node.data, input)[code])
            @inbounds node = node.children[code]
		end
        total_prob+=word_prob
    end
    total_prob
end


end #Module
