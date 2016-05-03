"""Perform Word Sense Disabmiguation, by chosing the word-sense that says the context words are most likely.
i.e. use the language modeling task.
Returns integer coresponding to to the Column of the embedding matrix for that word, for the best word sense embedding.
"""
function WSD(embed::WordSenseEmbedding, word::AbstractString, context::AbstractVector{AbstractString})
	sense_embeddings = embed.embedding[word]
	if length(sense_embeddings==1)
		return sense_embeddings[1]
	else
		prop, most_likely_sense_id = findmax([prob_of_context(embed, context, input) for input in sense_embeddings])
		return most_likely_sense_id
	end
end

"""
Returns a Vector of vectors for the movement of every point, given the forces vector of vectors
If no break occurs then there will be only one motion returned
"""
function get_motions{N<:AbstractFloat}(forces::Vector{Vector{N}}, strength)

    ndims=length(forces[1] )
    nforces = length(forces)
    motions = [Vector{N}(ndims) for _ in 1:nforces]    
#	[@assert(length(ff)==ndims) for ff in forces]

    for dim in 1:ndims
        north_force = 0.0
        south_force = 0.0
		n_north =0
		n_south =0
        for f_ii in 1:nforces
            force = forces[f_ii][dim]
            if force>0.0
                north_force+=force
				n_north+=1
            else
                south_force+=force
				n_south+=1
            end
        end
        tension = 2*min(north_force,-1*south_force) 
			#This is the resisted force. It that can break, rather than move. 
        if tension>strength
            for f_ii in 1:nforces
                force = forces[f_ii][dim]
                if force>0.0
                    motions[f_ii][dim] = north_force/n_north 
					#Don't actually decrease distence moved because of force resisted,
					#that is a metaphore.
                else
                    motions[f_ii][dim] = south_force/n_south 
                end
            end 
        else
            #does not break, so all forces apply same motion
            for f_ii in 1:nforces
                motions[f_ii][dim] = (north_force+south_force)/(n_north+n_south) 
            end          
        end
    end

    #Find unique motion rows, and stack up each occurrence.
    #These corespond to all the new points
	counts = Dict{Vector{N},Int64}()
	for motion in motions
#		@assert length(motion)==ndims
		get!(counts,motion,0)
		counts[motion]+=1
	end
	Vector{N}[motion.*count for (motion, count) in counts]
end


function break_and_move!(embed, word, sense_id, pending_forces)
	motions = get_motions(pending_forces,embed.strength)

	old_position = embed.embedding[word][sense_id]
	new_positions = [motion+old_position for motion in motions]
	splice!(embed.embedding[word],sense_id,new_positions) #Delete Old, insert new
	return embed.embedding[word]
end

"Given a window, actually does the training on it"
function train_window!(embed::WordSenseEmbedding, window::Vector{AbstractString},middle::Int64, α::AbstractFloat)
	word=window[middle]
	if !haskey(embed.codebook, word) #Not a word we are training, move on
		return embed
	end
	
	local_lsize = rand(0:embed.lsize)
	local_rsize = rand(0:embed.rsize)

	context = sub(window,[1:middle-1; middle+1:length(window)]) #IDEA: Should the local_lsize and local_rsize be used to find the context for WSD?
	sense_id = WSD(embed, word, context)


	#Make space to store the forces
	for s_ii in length(embed.pending_forces[word]):sense_id-1
		blank_forces=Vector{Float32}[]
		sizehint!(blank_forces, embed.force_minibatch_size)
		push!(embed.pending_forces[word],blank_forces)
	end

	pending_forces = embed.pending_forces[word][sense_id]
	input = embed.embedding[word][sense_id] 

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
			blank_forces=Vector{Float32}[]
			sizehint!(blank_forces, embed.force_minibatch_size)
			embed.pending_forces[word][sense_id] = blank_forces 
		end	
		
	end
	#TODO: Force all things to break_break_and_move! at end of run_training!
	embed
end


function initialize_embedding(embed::WordSenseEmbedding, ::RandomInited)
    for word in embed.vocabulary
        embed.embedding[word] = [rand(Float32,embed.dimension) * 2 - 1]
		embed.pending_forces[word] = Vector{Vector{Vector{Float32}}}()
    end
    embed
end
