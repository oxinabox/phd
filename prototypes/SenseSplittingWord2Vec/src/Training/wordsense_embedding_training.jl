"""Perform Word Sense Disabmiguation, by chosing the word-sense that says the context words are most likely.
i.e. use the language modeling task.
Returns integer coresponding to to the Column of the embedding matrix for that word, for the best word sense embedding.
"""
function WSD(embed::WordSenseEmbedding, word::AbstractString, context::Vector{AbstractString})
	embeddings = size(embed.embedding[word])
	most_likely_sense_id = -1
	max_prob=-Inf
	for sense_id in 1:size(embed.embedding[word],2)
		input = embeddings[sense_id]
		prob = prob_of_context(embed, context, input)
		if prob>max_prob
			max_prob=prob
			most_likely_sense_id = sense_id
		end
	end
	return most_likely_sense_id
end

"""
Returns a Vector of vectors for the movement of every point, given the forces vector of vectors
If no break occurs then there will be only one motion returned
"""
function get_motions(forces::Vector{Vector{Float32}}, strength)

    ndims = length(forces[1])
    nforces = length(forces)
    motions = map(similar,forces)    
    
    for dim in 1:ndims
        north_force = 0.0
        south_force = 0.0
        for f_ii in 1:nforces
            force = forces[f_ii][dim]
            if force>0.0
                north_force+=force
            else
                south_force+=force
            end
        end
        tension = 2*min(north_force, south_force) #This is the resisted force. It that can break, rather than move. 
        if tension>strength
            for f_ii in 1:nforces
                force = forces[f_ii]
                if force>0.0
                    motion[f_ii][dim] = (north_force-strength) 
                else
                    motion[f_ii][dim] = (strength-south_force) 
                end
            end 
        else
            #does not break, so all forces apply same motion
            for f_ii in 1:nforces
                motions[f_ii][dim] = (north_force-strength)                 
            end          
        end
    end
    #Find unique motion rows.
    #These corespond to all the new points
    unique(motions)
end


function break_and_move!(embed, word, sense_id, pending_forces)
	motions = get_motions(pending_forces,embed.strength)

	old_position = embed.embeddings[word][sense_id]
	embed.embeddings[word] = embed.embeddings[word][0:sense_id-1:sense_id+1:end] #remove old sense vectors
	for motion in motions
		push!(embed.embeddings[word], motion+old_position)
	end
	return embed.embeddings[word]
end

"Given a window, actually does the training on it"
function train_window!(embed::WordSenseEmbedding, window::Vector{AbstractString},middle::Int64, α::AbstractFloat)
	trained_word=window[middle]
	if !haskey(embed.codebook, trained_word) #Not a word we are training, move on
		return embed
	end
	
	local_lsize = rand(0: embed.lsize)
	local_rsize = rand(0: embed.rsize)

	context = sub(window,1:middle-1:middle+1:length(window)) #IDEA: Should the local_lsize and local_rsize be used to find the context for WSD?
	sense_id = WSD(embed, trained_word, context)


	#Make space to store the forces
	for s_ii in length(embed.pending_forces[word]):sense_id-1
		blank_forces=[Vector{Float32}()]
		sizehint!(blank_forces, embed.force_minibatch_size)
		push!(embed.pending_forces[word],blank_forces)
	end

	pending_forces = embed.pending_forces[word][sense_id]
	input = sub(embed.embedding[trained_word],sense_id) 	#Must changing inplace

	for ind in (middle - local_lsize) : (middle + local_rsize)
		(ind == middle) && continue

		target_word = window[ind]
		# discard words not presenting in the classification tree
		haskey(embed.codebook, target_word) || continue

		node = embed.classification_tree::TreeNode

		force = zeros(Float32, embed.dimension)

		for code in embed.codebook[target_word]
			train_one!(node.data, input, code, force, α)
			node = node.children[code]
		end
		push!(pending_forces,force)
		if length(pending_forces)>=embed.force_minibatch_size
			break_and_move!(embed, word, sense_id, pending_forces)

			blank_forces=[Vector{Float32}()]
			sizehint!(blank_forces, embed.force_minibatch_size)
			embed.pending_forces[word][sense_id] = blank_forces 
		end	
		
	end
	#TODO: Force all things to break_break_and_move! at end of run_training!
	embed
end


function initialize_embedding(embed::WordSenseEmbedding, randomly::RandomInited)
    for word in embed.vocabulary
        embed.embedding[word] = [rand(Float32,embed.dimension) * 2 - 1]
		embed.pending_forces[word] = Vector{Vector{Vector{Float32}}}()
    end
    embed
end
