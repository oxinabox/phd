
"Given a window, actually does the training on it"
function train_window!(embed::WordEmbedding, window::Vector{AbstractString},middle::Int64, α::AbstractFloat)
	trained_word=window[middle]
	
	local_lsize = rand(0: embed.lsize)
	local_rsize = rand(0: embed.rsize)
	
	input = embed.embedding[trained_word] #Inplace changing
    input_gradient=similar(input) #for spead preallocate then just change this one vector 
	for ind in (middle - local_lsize) : (middle + local_rsize)
		(ind == middle) && continue

		target_word = window[ind]
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


#Todo move this into the plain word embedding file.
"""Reterns the window, and the α, for this round of training.
Also logs the progress"""
function training_windows(embed::WordEmbedding,
						  stream::WordStream,
						  end_of_iter_callback::Function=identity)
	
	Task() do
		tic()
		α = embed.init_learning_rate
		trained_count = 0
		end_of_iter_callback((0,embed))
		for current_iter in 1:embed.iter
			debug("Iter $current_iter of $(embed.iter)")
            windows = RandomShuffler(sliding_window(stream, lsize=embed.lsize, rsize=embed.rsize), 1024)
			for window in windows
				trained_count += 1
				α = get_α_and_log(embed, trained_count, α)
				produce(window,α)
			end
			debug("Running Callback after $current_iter")
			end_of_iter_callback((current_iter,embed))
		end
		overall_time = toq()
		debug("Finished training. Trained on $(trained_count) words in $(overall_time) seconds.")
	end
end


"Runs all the training, handles adjusting learning rate, repeating through loops etc."
function run_training!(embed::WordEmbedding, 
					   words_stream::WordStream;
					   strip::Bool=false,
					   end_of_iter_callback::Function=identity)
	middle = embed.lsize + 1
    trained_times = Dict{AbstractString, Int64}()

	for (window, α) in training_windows(embed,words_stream,end_of_iter_callback)
		trained_word = window[middle]
		trained_times[trained_word] = get(trained_times, trained_word, 0) + 1
		train_window!(embed,window,middle,α)
    end

    embed.trained_times = trained_times

    # strip to remove unnecessary members and make serialization faster
    strip && keep_word_vectors_only!(embed)
    embed
end


function initialize_embedding(embed::WordEmbedding, randomly::RandomInited)
    for i in embed.distribution |> keys
        embed.embedding[i] = 0.1*rand(embed.dimension) * 2 - 1
    end
    embed
end

