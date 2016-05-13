import DataStructures.DefaultDict
using WorkIterator

"""Perform Word Sense Disabmiguation, by chosing the word-sense that says the context words are most likely.
i.e. use the language modeling task.
Returns integer coresponding to to the Column of the embedding matrix for that word, for the best word sense embedding.
"""
@inline function WSD(embed::WordSenseEmbedding, word::AbstractString, context::AbstractVector{AbstractString})
	sense_embeddings = embed.embedding[word]
	if length(sense_embeddings)==1
		return 1
	else
		prop, sense_id = findmax([logprob_of_context(embed, context, input) for input in sense_embeddings])
		return sense_id
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
		counts[motion]=1 + get!(counts,motion,0)
		#@assert(!(any(isnan(motion))))
	end
	Vector{N}[motion.*count for (motion, count) in counts]
end


#TODO Add α here.
function break_and_move!(embed::WordSenseEmbedding,pending_forces, word::AbstractString, sense_id::Int64)
	forces = pending_forces[word][sense_id]
	if length(forces)>0
		motions = get_motions(forces, embed.strength)

		old_position = embed.embedding[word][sense_id]
		new_positions = [motion+old_position for motion in motions]
		splice!(embed.embedding[word],sense_id,new_positions) #Delete Old, insert new
		pending_forces[word][sense_id] = blank_forces(embed.force_minibatch_size)
		#Delete processed forces
	end
	return embed.embedding[word]
end

"Apply break and move across all forces"
function break_and_move!(embed::WordSenseEmbedding, pending_forces)
	for word in keys(pending_forces) |> collect #HACK for some reason you can't directly iterate the keys
		for sense_id in keys(pending_forces[word]) |> collect
			break_and_move!(embed, pending_forces, word, sense_id)
		end
	end
	embed.embedding
end

function blank_forces(nforces_hint::Int64)
    blank_forces=Vector{Float32}[]
    sizehint!(blank_forces, nforces_hint)
    blank_forces
end


"Given a window, actually does the training on it"
function train_window!(embed::WordSenseEmbedding,pending_forces, context, word, sense_id , α::AbstractFloat)
    ws_pending_forces = pending_forces[word][sense_id]
	input = embed.embedding[word][sense_id] 

	for target_word in context 
		node = embed.classification_tree::TreeNode
		force = zeros(Float32, embed.dimension)
		for code in embed.codebook[target_word]
			train_one!(node.data, input, code, force, α)
			node = node.children[code]
		end
		push!(ws_pending_forces,force)
	end
	embed
end


@inline function WsdTrainingCase(embed::WordEmbeddings.WordSenseEmbedding, window)
    word = window[embed.lsize+1]
	context = window[[1:embed.lsize; embed.lsize+1:end]]
    sense_id = Training.WSD(embed, word, context)
    
    return (context, word, sense_id)
end


#HACK: lets debug what is being serialised by overloading the calls
function Base.serialize(s::Base.SerializationState, x::WordSenseEmbedding)
	tic()
	Base.Serializer.serialize_any(s,x)
	tt=toq()
	open("selog.txt","a") do fp
		println(fp, tt)
	end
end



"Runs all the training, handles adjusting learning rate, repeating through loops etc."
function run_training!(embed::WordSenseEmbedding, 
					   words_stream;
					   strip::Bool=false,
					   end_of_iter_callback::Function=identity)
	middle = embed.lsize + 1
    forces_for_sense = Vector{Vector{Float32}}
    sense_forces = ()->DefaultDict(Int64, forces_for_sense,
                                   ()->blank_forces(1024)) #TODO: Put a number here based on Word Distribution?
    pending_forces = DefaultDict(AbstractString,typeof(sense_forces()), sense_forces)

	
	debug("Running End of Iter callback, before first iter")
	end_of_iter_callback((0,embed))
	
	trained_count=0
	α=embed.init_learning_rate
	#PREMOPT: consider initially having just one worker, then adding 2 per iteration, as the  workload increases due to splitting
    for iter in 1:embed.iter
		windows = sliding_window(words_stream, lsize=embed.lsize, rsize=embed.rsize)
		
		for minibatch in Base.partition(windows, embed.force_minibatch_size)
			#cases = Base.pgenerate(default_worker_pool(), win->WsdTrainingCase(embed,win), minibatch)
			cases = _pgenerate_gh(win->WsdTrainingCase(embed,win), minibatch, :ss_wsdtrainingcase)
			for (context, word, sense_id) in cases
				trained_count+=1
				α = get_α_and_log(embed, trained_count, α)
				train_window!(embed, pending_forces, context,word, sense_id,α)
			end
			break_and_move!(embed,pending_forces)			
		end
		debug("Running End of Iter callback")
		end_of_iter_callback((iter,embed))
	end


    # strip to remove unnecessary members and make serialization faster
    strip && keep_word_vectors_only!(embed)
    embed
end


function initialize_embedding(embed::WordSenseEmbedding, ::RandomInited)
    for word in embed.distribution |> keys
        embed.embedding[word] = [rand(Float32,embed.dimension) * 2 - 1]
    end
    embed
end
