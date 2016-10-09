module AdaGramCompat
using Query
using AdaGram
const AdaGram_lib = joinpath(Pkg.dir("AdaGram"), "lib", "superlib.so")
import WordEmbeddings.all_word_sense_vectors

export word_sense_vectors, word_sense_vector, AdaGramModel

immutable AdaGramModel
    vm::AdaGram.VectorModel
    dict::AdaGram.Dictionary
end
AdaGram.disambiguate(am::AdaGramModel, word, context) = disambiguate(am.vm, am.dict, word, context)


function all_word_sense_vectors(am::AdaGramCompat.AdaGramModel, word)
    if haskey(am.dict.word2id, word)
        wsv_mat = word_sense_vectors(am, word)
        [view(wsv_mat,:,ii) for ii in 1:size(wsv_mat,2)]
    else
        Vector{Float32}[]
    end
end


function word_sense_vectors(am::AdaGramModel, word)
	view(am.vm.In, :, :, am.dict.word2id[word])
end

function word_sense_vector(am::AdaGramModel, word, sense_id::Int)
	view(word_sense_vectors(am,word), :, sense_id)
end

function Query.logprob_of_context(am::AdaGramModel, context, input::AbstractVector{Float32}; skip_oov=false, normalise_over_length=false)
    total_lprob=zero(Float32)
	context_length = 0
    for context_word in context
        skip_oov && !haskey(am.dict.word2id, context_word) && continue
        context_length+=1
        context_word_id = am.dict.word2id[context_word]
        context_word_path = view(am.vm.path, :, context_word_id)
        context_word_code = view(am.vm.code, :, context_word_id)

        word_lprob = ccall((:skip_gram, AdaGram_lib), Float32,
            (Ptr{Float32}, Ptr{Float32}, Clonglong, Ptr{Int32}, Ptr{Int8},        Clonglong),
            input, am.vm.Out,  size(am.vm.In, 1), context_word_path, context_word_code, size(am.vm.code, 1))
        total_lprob+=word_lprob
    end

	if context_length==0
		throw(Query.NoContextError(context))
	end

    if normalise_over_length
        total_lprob/=context_length #This is equivlent to taking the context_length-th root in nonlog domain. Which makes sense.
	end
    total_lprob
end


end #module
