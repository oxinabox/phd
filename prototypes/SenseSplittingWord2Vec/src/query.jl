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
