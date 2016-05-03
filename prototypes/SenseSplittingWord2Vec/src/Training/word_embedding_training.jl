
"Given a window, actually does the training on it"
function train_window!(embed::WordEmbedding, window::Vector{AbstractString},middle::Int64, α::AbstractFloat)
	trained_word=window[middle]
	if !haskey(embed.codebook, trained_word)
		return embed
	end
	
	local_lsize = rand(0: embed.lsize)
	local_rsize = rand(0: embed.rsize)
	
	input = embed.embedding[trained_word] #Inplace changing
    input_gradient=similar(input) #for spead preallocate then just change this one vector 
	for ind in (middle - local_lsize) : (middle + local_rsize)
		(ind == middle) && continue

		target_word = window[ind]
		# discard words not presenting in the classification tree
		haskey(embed.codebook, target_word) || continue

		node = embed.classification_tree::TreeNode

		fill!(input_gradient, 0.0)

		for code in embed.codebook[target_word]
			train_one!(node.data, input, code, input_gradient, α)
			node = node.children[code]
		end
		for ii in 1:embed.dimension
			input[ii] -= input_gradient[ii]
		end
	end
	embed
end

function initialize_embedding(embed::WordEmbedding, randomly::RandomInited)
    for i in embed.vocabulary
        embed.embedding[i] = rand(embed.dimension) * 2 - 1
    end
    embed
end

