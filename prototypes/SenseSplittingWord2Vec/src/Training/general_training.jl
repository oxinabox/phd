#*===============================================================================
#* Calculate Word Embedding
#*Train a corpus
#*Step 1: Find Word Distribution
#*Step 2: Calculate Word Embedding
#*===============================================================================#

############ Callbacks
function save_callback(basefilename::String)
	function scallbck(arg)
		iter, embed = arg
		
		save(embed, "$(basefilename)_i$(iter).model")
	end

end



#######################

function get_α_and_log(embed::GenWordEmbedding, trained_count, α)
	if trained_count % 10000 == 0
		progress = trained_count/ (embed.iter*embed.corpus_size)
		info(string(round(progress*100,2))*"% - trained on $trained_count words", progress=progress, α=α)
		α = embed.init_learning_rate * (1 - progress)
		if α < embed.init_learning_rate * 0.0001
			α = embed.init_learning_rate * 0.0001
		end
	end
	α
end



#Todo move this into the plain word embedding file.
"""Reterns the window, and the α, for this round of training.
Also logs the progress"""
function training_windows(embed::GenWordEmbedding,
						  stream::WordStream,
						  end_of_iter_callback::Function=identity)
	
	Task() do
		tic()
		α = embed.init_learning_rate
		trained_count = 0
		end_of_iter_callback((0,embed))
		for current_iter in 1:embed.iter
			debug("Iter $current_iter of $(embed.iter)")
			windows = sliding_window(stream, lsize=embed.lsize, rsize=embed.rsize)
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




function initialize_network(embed::GenWordEmbedding, huffman::HuffmanTree)
    heap = PriorityQueue()
    for (word, freq) in embed.distribution
        node = BranchNode([], word)    # the data field of leaf node is its corresponding word.
        enqueue!(heap, node, freq)
    end
    while length(heap) > 1
        (node1, freq1) = Base.Collections.peek(heap)
        dequeue!(heap)
        (node2, freq2) = Base.Collections.peek(heap)
        dequeue!(heap)
        newnode = BranchNode([node1, node2], LinearClassifier(2, embed.dimension)) 
								#the data field of internal node is the classifier
        enqueue!(heap, newnode, freq1 + freq2)
    end
    embed.classification_tree = dequeue!(heap)
    embed
end


function train(embed::GenWordEmbedding, corpus_filename::AbstractString;
			   end_of_iter_callback::Function=identity)

    embed.distribution, embed.corpus_size = word_distribution(corpus_filename)
    embed.vocabulary = collect(keys(embed.distribution))

    initialize_embedding(embed, embed.init_type)        # initialize by the specified method
    initialize_network(embed, embed.network_type)

    # determine the position in the tree for every word
    for (w, code) in leaves_of(embed.classification_tree)
        embed.codebook[w] = code
    end

    t1 = time()
    println("Starting sequential training...")
    words_stream = words_of(corpus_filename, subsampling = (embed.subsampling, true, embed.distribution))
    run_training!(embed, words_stream, end_of_iter_callback=end_of_iter_callback)

    t2 = time()
    println("Training complete at $(t2-t1) time")
    embed
end



