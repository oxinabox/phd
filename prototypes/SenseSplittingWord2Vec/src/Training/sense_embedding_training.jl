import DataStructures.DefaultDict
using  MultivariateStats


################ Callbacks
export sense_counts_callback
import JSON
function sense_counts_callback(savename)
	function callback(arg)
		meta = arg[1]
		embed::WordSenseEmbedding=arg[2]
		open(savename*"_counts.json","a") do fp
			JSON.print(fp, (meta,Dict([word=>length(sense_embeds) for (word, sense_embeds) in embed.embedding])))
			println(fp)       
		end
	end
end

#################


"""Perform Word Sense Disabmiguation, by chosing the word-sense that says the context words are most likely.
i.e. use the language modeling task.
Returns integer coresponding to to the Column of the embedding matrix for that word, for the best word sense embedding.
"""
@inline function WSD{S<:String}(embed::WordSenseEmbedding, word::String, context::AbstractVector{S}; skip_oov=false)
	sense_embeddings = embed.embedding[word]
	if length(sense_embeddings)==1
		return 1
	else
		prob, sense_id = findmax([logprob_of_context(embed, context, input; skip_oov=skip_oov) for input in sense_embeddings])
		return sense_id
	end
end


"Returns a vector of BitVector, each bitvector corresponds to a unique splitting direction"
@inline function get_directions(forces::Matrix, strength::Number)
    ndims, nforces=size(forces)
    directions = [BitVector(ndims) for _ in 1:nforces]
    #False is North split, or nosplit
    
    @inbounds for dim in 1:ndims
        north_force = 0.0
        south_force = 0.0
        @inbounds for f_ii in 1:nforces
            force = forces[dim,f_ii]
            if force>0.0
                north_force+=force
            else
                south_force+=force
            end
        end
        tension = 2*min(north_force,-1*south_force) 
			#This is the resisted force. It that can break, rather than move. 
        if tension>strength*nforces
            @inbounds for f_ii in 1:nforces
                force = forces[dim,f_ii]
                directions[f_ii][dim]=force<0.0
            end 
        end
    end
    directions
end

"""
Returns a Vector of vectors for the movement of every point, given the forces vector of vectors
If no break occurs then there will be only one motion returned
"""
@fastmath function get_motions{N<:AbstractFloat}(forces::Vector{Vector{N}}, strength::Number; pca_kwargs...)
    nforces = length(forces)
    forces_mat = hcat(forces...)
	try	
		@assert(all(isfinite(forces_mat)), "Nonfinite Forces Matrix with elements: $(forces_mat[!isfinite(forces_mat)])")
		dim_reducer=fit(PCA, forces_mat; pca_kwargs...) 
		#Setting the mean as zero, indicating it is already centered, 
		#so PCA will not recenter it, which could change directions
		#if the forces are of very different magnitude eg [1,0f0],[4,0f0], with strength 0.5 results on two forces.
		#TODO: Consider if this is not infact a good thing
		red_forces = transform(dim_reducer, forces_mat)
			
		directions = get_directions(red_forces, strength)
		#Find unique motion rows, and stack up each occurrence.
		#These corespond to all the new points
		motions = Dict{BitVector,Vector{N}}()
		@inbounds for f_ii in 1:nforces
			direction = directions[f_ii]
			force = forces[f_ii]
			
			if haskey(motions, direction)
				motions[direction]+=force
			else
				motions[direction]=force
			end
		end
		#println(map(bits, keys(motions)))
		collect(values(motions))
	catch
		open("ErrorMat_time"*string(time())*".jsz","w") do fp
			serialize(fp,forces_mat)
		end
		rethrow()
	end
end


function break_and_move!(word_sense_embeddings,pending_forces_word, strength::Number, nsplitaxes::Integer)
	scaled_strength = strength * 2^(length(word_sense_embeddings)-1)  #More embeddings it has, the harder is it to create more.
	for sense_id in keys(pending_forces_word) |> collect
		forces = pending_forces_word[sense_id]
		if length(forces)>0			
            motions = get_motions(forces, scaled_strength; maxoutdim=nsplitaxes)
			old_position = word_sense_embeddings[sense_id]
			new_positions = [motion+old_position for motion in motions]
			splice!(word_sense_embeddings, sense_id, new_positions) #Delete Old, insert new
		end
	end
	word_sense_embeddings
end


"Apply break and move across all forces"
function break_and_move!(embed::SplittingWordSenseEmbedding, pending_forces)
	#What this actually Does, if not for need to marshal interprocess communication
	#@sync for word in keys(pending_forces) |> collect
	#	@async embed.embedding[word] = remote(break_and_move!)(embed.embedding[word],
	#															pending_forces[word],
	#															embed.strength)
	#end

	word_args = Task() do
		for (word,pending_forces_word) in pending_forces
            produce(word, embed.embedding[word], pending_forces_word, embed.strength, embed.nsplitaxes)
		end
	end

	returned_embeddings = _pgenerate_gh(word_args, :break_and_move) do args
		(args[1], break_and_move!(args[2:end]...))
	end

	for (word,word_embeddings) in returned_embeddings
		embed.embedding[word] = word_embeddings
	end
	embed.embedding
end


"Given a window, actually does the training on it"
function train_window!{S<:String}(embed::SplittingWordSenseEmbedding, pending_forces, context::AbstractVector{S}, word::S, sense_id::Integer, α::AbstractFloat)
	input = embed.embedding[word][sense_id] 
	@assert(all(abs(input).<10.0^10.0))
	
	total_grad = zeros(Float32, embed.dimension)
	try
		for target_word in context 
			node = embed.classification_tree::TreeNode
			for code in embed.codebook[target_word]
				train_one!(node.data, input, code, total_grad, α)
				node = node.children[code]
			end
		end
		total_force = -total_grad #Force is in opposite direction to gradient
		all(isfinite(total_force)) || throw(InvalidStateException("Nonfinite force Produced: $(total_force[!isfinite(total_force)])", :NonFiniteForce))
		push!(pending_forces[word][sense_id], total_force)
		embed
	catch
		open("err_train_window$(time()).jsz", "w") do fp
			serialize(fp, Dict(
				:embed=>embed,
				:pending_forces=>pending_forces,
				:context=>context,
				:word=> word,
				:sense_id=>sense_id,
			))
		end	
		rethrow()
	end	
end


@inline function WsdTrainingCase(embed::WordEmbeddings.WordSenseEmbedding, window)
	w_ind = embed.lsize+1
    word = window[w_ind]
	#Dynamic Window #NOTE: Does it matter that this is not symetric? I think not
	pre_words =  rand(1:embed.lsize-1)
	post_words = w_ind+1 + rand(1:embed.rsize-1)

	context = window[[pre_words:w_ind-1; w_ind+1:post_words]]
    sense_id = Training.WSD(embed, word, context)
    
    return (context, word, sense_id)
end

function blank_pending_forces()
	forces_for_sense = Vector{Vector{Float32}}
    sense_forces = ()->DefaultDict(Int64, forces_for_sense,
                                   ()->sizehint!(Vector{Float32}[], 1024)
								   #TODO: Put a size hint number here based on Word Distribution?
								   )
    pending_forces = DefaultDict(String,typeof(sense_forces()), sense_forces)
end


"Runs all the training, handles adjusting learning rate, repeating through loops etc."
function run_training!(embed::SplittingWordSenseEmbedding, 
					   words_stream;
					   end_of_iter_callback::Function=identity,
					   end_of_minibatch_callback::Function=identity,
					   )
    	
	debug("Running End of Iter callback, before first iter")
	end_of_iter_callback((0,embed))
	
	trained_count=0
	α=embed.init_learning_rate
    for iter in 1:embed.iter
		windows = sliding_window(words_stream, lsize=embed.lsize, rsize=embed.rsize)
		
		for minibatch in Base.partition(windows, embed.force_minibatch_size)
			pending_forces = blank_pending_forces()
			#cases = Base.pgenerate(default_worker_pool(), win->WsdTrainingCase(embed,win), minibatch)
			cases = _pgenerate_gh(win->WsdTrainingCase(embed,win), minibatch, :ss_wsdtrainingcase)
            for (context, word, sense_id) in ReservoirShuffler(cases,1024)
				trained_count+=1
				α = get_α_and_log(embed, trained_count, α)
				train_window!(embed, pending_forces, context,word, sense_id,α)
			end
			break_and_move!(embed,pending_forces)
			end_of_minibatch_callback((trained_count,embed))
		end
		debug("Running End of Iter callback")
		end_of_iter_callback((iter,embed))
	end
    embed
end


function initialize_embedding(embed::WordSenseEmbedding, ::RandomInited)
    for word in embed.distribution |> keys
        embed.embedding[word] = [rand(Float32,embed.dimension) * 2 - 1]
    end
    embed
end
