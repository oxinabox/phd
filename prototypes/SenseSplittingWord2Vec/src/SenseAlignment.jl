module SenseAlignment

using WordNet
using AdaGram
using AdaGramCompat
using WordEmbeddings, SoftmaxClassifier
using Utils
using Query
using Distances


export normal_probs, general_wsd, synthesize_embedding

normal_probs(logprobs::Vector) = normal_probs(copy(logprobs))
function normal_probs!{F<:AbstractFloat}(logprobs::Vector{F})
    ret = logprobs
    max_lp = maximum(logprobs)
    ret.-=max_lp #Bring closer to zero
    map!(exp,ret)
    denom = sum(ret)
    ret./=denom
    ret
end

function general_wsd(ee, context, wvs, priors=ones(length(wvs)), normalise_over_context_length::Bool=true)
    lps = Vector{Float64}(length(wvs))
    for (ii,(prior, wv)) in enumerate(zip(priors, wvs))
        lps[ii] = Query.logprob_of_context(ee, context, wv; skip_oov=true, normalise_over_length=normalise_over_context_length)
        lps[ii] += log(prior)
    end
    normal_probs!(lps)
end

function synthesize_embedding(ee,context::AbstractVector,
                               word_or_phrase::AbstractString,
                               fallback_word_or_phrase=""::AbstractString)
    
    words = split(word_or_phrase, " ")
    wvs = vcat((all_word_sense_vectors(ee,w) for w in words)...)
    if length(wvs) == 0
        fallbacks = split(fallback_word_or_phrase, " ")
        wvs = vcat((all_word_sense_vectors(ee,w) for w in fallbacks)...)
        if length(wvs) == 0
            throw(KeyError(" $words, nor $fallbacks have embeddings"))
        end
    end

    probs = general_wsd(ee, context, wvs)
    sum(wvs.*probs)
end




end #module
