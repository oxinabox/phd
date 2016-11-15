module SenseAlignment

using WordNet
using AdaGram
using AdaGramCompat
using WordEmbeddings, SoftmaxClassifier
using Utils
using Query
using Distances


export normal_probs, normal_probs!, general_wsd, synthesize_embedding



normal_probs(logprobs::Vector) = normal_probs!(copy(logprobs))
function normal_probs!{F<:AbstractFloat}(logprobs::Vector{F})
    ret = logprobs
    max_lp = maximum(logprobs)
    ret.-=max_lp #Bring closer to zero
    map!(exp,ret)
    denom = sum(ret)
    ret./=denom
	@assert(sum(ret) ≈ one(F), "Probabilities don't sum to 1, instead are requal to $(sum(ret)), from $ret")
    ret
end

function general_wsd(ee, context, wvs, priors=ones(length(wvs));
					 normalise_over_context_length::Bool=true,
					 normalize_over_prior::Bool=false
					 )
	if normalize_over_prior && !normalise_over_context_length
		error("Not set to normalise over context length, so can not normalise over prior")
	end
	#TODO: Please spell normalize consistently

	lps = Vector{Float64}(length(wvs))

	context_length = sum(has_word(ee, context_word)
						 for context_word in context)

    for (ii,(prior, wv)) in enumerate(zip(priors, wvs))
        lps[ii] = logprob_of_context(ee, context, wv; skip_oov=true)

		if normalise_over_context_length && !normalize_over_prior
			lps[ii]/=context_length
		end

		@assert(!isnan(lps[ii]))
		lps[ii] += log(prior)

		if normalise_over_context_length && normalize_over_prior
			lps[ii]/=context_length
		end
    end
    normal_probs!(lps)
end




function WordEmbeddings.all_word_sense_vectors(ee,
			word_or_phrase::AbstractString,
			fallback_word_or_phrase::AbstractString)
	words = split(word_or_phrase, " ")
	wvs = vcat((all_word_sense_vectors(ee,w) for w in words)...)
	if length(wvs) == 0
		fallbacks = split(fallback_word_or_phrase, "_")
		wvs = vcat((all_word_sense_vectors(ee,w) for w in fallbacks)...)
		if length(wvs) == 0
			throw(KeyError(" $words, nor $fallbacks have embeddings"))
		end
	end
	wvs
end

function WordEmbeddings.all_word_sense_priors(ee,
			word_or_phrase::AbstractString,
			fallback_word_or_phrase::AbstractString)
	words = split(word_or_phrase, " ")
	wvs = vcat((all_word_sense_priors(ee,w) for w in words)...)
	if length(wvs) == 0
		fallbacks = split(fallback_word_or_phrase, "_")
		wvs = vcat((all_word_sense_priors(ee,w) for w in fallbacks)...)
		if length(wvs) == 0
			throw(KeyError(" $words, nor $fallbacks have priors"))
		end
	end
	wvs
end



function synthesize_embedding(ee,context::AbstractVector,
                               word_or_phrase::AbstractString,
                               fallback_word_or_phrase::AbstractString="";
							   use_prior::Bool = false,
							   normalise_over_context_length::Bool=true,
							   normalize_over_prior::Bool=false
							   )
	wvs = all_word_sense_vectors(ee, word_or_phrase, fallback_word_or_phrase)
	prior = if use_prior
		all_word_sense_priors(ee, word_or_phrase, fallback_word_or_phrase)
	else
		ones(Float32,length(wvs))
	end

	probs = general_wsd(ee, context, wvs, prior;
			normalise_over_context_length=normalise_over_context_length,
			normalize_over_prior=normalize_over_prior
			)
    sum(wvs.*probs)
end


end #module
