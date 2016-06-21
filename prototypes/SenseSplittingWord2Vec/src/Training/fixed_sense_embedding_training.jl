

"Given a window, actually does the training on it"
function train_window!{S<:String}(embed::FixedWordSenseEmbedding, context::AbstractVector{S}, word::S, sense_id::Integer, α::AbstractFloat)
	input = embed.embedding[word][sense_id] 
	@assert(all(abs(input).<10.0^10.0))
	embed.trained_times[word][sense_id]+=1	
	try
		for target_word in context
			input_grad = zeros(Float32, embed.dimension)
			node = embed.classification_tree::TreeNode
			for code in embed.codebook[target_word]
				train_one!(node.data, input, code, input_grad, α)
				node = node.children[code]
			end
			input[:] -= input_grad[:]	
		end
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



"Runs all the training, handles adjusting learning rate, repeating through loops etc."
function run_training!(embed::FixedWordSenseEmbedding, 
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
		info("Begin Iter $iter")	
		for minibatch in Base.partition(windows, embed.force_minibatch_size)
			cases = _pgenerate_gh(win->WsdTrainingCase(embed,win), minibatch, :ss_wsdtrainingcase)
			for (context, word, sense_id) in ReservoirShuffler(cases,64)
				trained_count+=1
				α = get_α_and_log(embed, trained_count, α)
				train_window!(embed, context,word, sense_id,α)
			end
			debug("Running End of Minibatch callback")
			end_of_minibatch_callback((trained_count,embed))
		end
		debug("Running End of Iter callback")
		end_of_iter_callback((iter,embed))
	end
    embed
end


function initialize_embedding(embed::FixedWordSenseEmbedding, ::RandomInited)
    for (word,freq) in embed.distribution
		nsenses = freq*embed.corpus_size >= embed.min_count_for_multiple_senses ? embed.initial_nsenses : 1
        embed.embedding[word] = [rand(Float32,embed.dimension) * 2 - 1 for _ in 1:nsenses]
    end
    embed
end
