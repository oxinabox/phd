module Query
using Base.Collections
using WordEmbeddings
using Distances
using SoftmaxClassifier
using NearestNeighbors
import NearestNeighbors.NNTree
export find_nearest_words, logprob_of_context, nn_tree, WSD, angular_dist, AngularDist


########### nearest_words, and analogy math
"The metric equivelent of Cosine Distence"
immutable AngularDist<:Metric
end
function Distances.evaluate(::AngularDist, a::AbstractArray, b::AbstractArray)
    cdist = cosine_dist(a,b)
    cos_val = 1-cdist
    acos(cos_val)/π
end
angular_dist(a::AbstractArray, b::AbstractArray) = evaluate(AngularDist(), a, b)

function nn_tree(embed::WordSenseEmbedding, metric::Metric=AngularDist())
    num_embeddings = sum(map(length,values(embed.embedding)))
	labels = Vector{Tuple{typeof(first(embed.embedding)[1]),Int64}}(num_embeddings)
	points = Matrix{eltype(first(embed.embedding)[2][1])}((embed.dimension, num_embeddings))
	ii = 0
	for (word,sense_embeddings) in embed.embedding
		for (sense_id, sense_embedding) in enumerate(sense_embeddings)
			ii+=1
			@inbounds labels[ii]=(word,sense_id)
			@inbounds points[:,ii]=sense_embedding
		end
	end
    dtree = BallTree(points, metric)
    (dtree,labels)
end


function nn_tree(embed::WordEmbedding, metric::Metric=AngularDist())
	nn_tree(embed.embedding, metric)
end

function nn_tree{K,V}(embeddings::Associative{K,V}, metric::Metric=AngularDist())
    num_embeddings = length(embeddings)
	dim_embeddings = length(first(embeddings)[1])
	labels = Vector{K}(num_embeddings)
	points = Matrix{eltype(V)}((dim_embeddings, num_embeddings))
	
	ii = 0
	for (word,embedding) in embeddings
		ii+=1
		@inbounds labels[ii]=word
		@inbounds points[:,ii]=embedding
	end
    dtree = BallTree(points, metric)
    (dtree,labels)
end

function find_nearest_words(embed::WordEmbedding, equation::String; nwords=5)
	dtree,labels= nn_tree(embed)
	find_nearest_words(embed, equation, dtree,labels, nwords=nwords)
end

function find_nearest_words(embed::WordEmbedding, equation::String, dtree,labels; nwords=5)
	tokens = replace(replace(equation, "+", " + "), "-", " - ")
    positive_words = String[]
    negative_words = String[]
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
	
	find_nearest_words(embed, positive_words, negative_words,dtree,labels; nwords=nwords)
end

find_nearest_words(embed::WordEmbedding, positive_words::Vector, negative_words::Vector; nwords=5) = find_nearest_words(embed, positive_words, negative_words,nn_tree(embed); nwords=nwords)
function find_nearest_words(embed::WordEmbedding, positive_words::Vector, negative_words::Vector, dtree,labels; nwords=5)
    wv = sum([embed.embedding[w] for w in positive_words])
	wv .-= length(negative_words)>0 ? sum([embed.embedding[w] for w in negative_words]) : 0.0

	find_nearest_embedding(dtree,labels, wv; nwords=nwords, banned=[positive_words;negative_words]) 
end


find_nearest_words(embed::WordSenseEmbedding, word, sense_id; nwords=5) = find_nearest_words(embed, word, sense_id, nn_tree(embed)...; nwords=nwords)
function find_nearest_words(embed::WordSenseEmbedding, word, sense_id, dtree,labels; nwords=5)
	wv = embed.embedding[word][sense_id]
	find_nearest_embedding(dtree, labels, wv; nwords=nwords, banned=[(word,sense_id)]) 
end


find_nearest_words(embed::WordSenseEmbedding, word; nwords=5) = find_nearest_words(embed, word,nn_tree(embed)...; nwords=nwords)
function find_nearest_words(embed::WordSenseEmbedding, word, dtree,labels; nwords=5)
	wvs = hcat(embed.embedding[word]...)
	find_nearest_embedding(dtree, labels, wvs; nwords=nwords,
							banneds=[(word,si) for si in 1:length(embed.embedding[word])]) 
end



function _gather_nearests(nwords, idxs, dists, labels, banned)
	ret = Vector{Tuple{eltype(labels),eltype(dists)}}(nwords)
	ii=0
	for (id, dist) in zip(idxs,dists)
		labels[id] ∈ banned && continue
		ii+=1
		ii>nwords && break
		ret[ii]=(labels[id],dist)
	end
	ret	
end

function find_nearest_embedding(embeddings_dtree::NNTree, labels, target_embedding::Vector; nwords=5, banned=[])
	max_words = nwords+length(banned)
	max_words = min(max_words,length(labels))
	nwords = min(nwords,max_words)
	idxs, dists = knn(embeddings_dtree,target_embedding, max_words,true)
	_gather_nearests(nwords, idxs, dists, labels, banned)
end


function find_nearest_embedding(embeddings_dtree::NNTree, labels, target_embeddings::Matrix; nwords=5, banneds=[[]])
	max_words = nwords+maximum(map(length,banneds))
	max_words = min(max_words,length(labels))
	nwords = min(nwords,max_words)
	if length(banneds)==1
		banneds = repeated(banneds)
	end

	idxss, distss = knn(embeddings_dtree,target_embeddings, max_words,true)
	[_gather_nearests(nwords, idxs, dists, labels, banned) for (idxs, dists, banned) in zip(idxss,distss, banneds)]
end



#################### Probability of the context
    
function logprob_of_context{S<:String}(embed::WordEmbedding, context::AbstractVector{S}, middle_word::S; kwargs...)
    input = embed.embedding[middle_word]
    logprob_of_context(embed, context, input; kwargs...)
end


function logprob_of_context(embed::GenWordEmbedding, context, input::Vector{Float32}; skip_oov=false, normalise_over_length=false)
    total_prob=0.0f0
	context_length = 0
    for target_word in context
		skip_oov && !haskey(embed.codebook, target_word) && continue
        context_length+=1

		node = embed.classification_tree      
        word_prob = 0.0f0
        for code in embed.codebook[target_word]  
            word_prob+= log(predict(node.data, input)[code])
            @inbounds node = node.children[code]
		end
        total_prob+=word_prob
    end
    if normalise_over_length
		total_prob/=context_length #This is equivlent to taking the context_length-th root in nonlog domain. Which makes sense.
	end
	total_prob::Float32
end


"""Perform Word Sense Disabmiguation, by chosing the word-sense that says the context words are most likely.
i.e. use the language modeling task.
Returns integer coresponding to to the Column of the embedding matrix for that word, for the best word sense embeddi
ng.
"""
@inline function WSD{S<:AbstractString}(embed::WordSenseEmbedding, word::String, context::AbstractVector{S}; skip_oov=false)
    sense_embeddings = embed.embedding[word]
	WSD(embed, sense_embeddings, context; skip_oov=skip_oov)
end

@inline function WSD{S<:AbstractString}(embed::WordSenseEmbedding, sense_embeddings, context::AbstractVector{S}; skip_oov=false)
    if length(sense_embeddings)==1
        return 1
    else
        #NOTE: Using  Experimental Threading
        lps = Vector{Float32}(length(sense_embeddings))
        Threads.@threads for ii in 1:length(sense_embeddings)
            input = sense_embeddings[ii]
            lps[ii] = logprob_of_context(embed, context, input; skip_oov=skip_oov)
        end
        prob, sense_id = findmax(lps)
        return sense_id
    end
end



end #Module
