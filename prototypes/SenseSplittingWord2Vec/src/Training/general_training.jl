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


function train(embed::GenWordEmbedding, corpus_filename::AbstractString; kwargs...)

    embed.distribution, embed.corpus_size = word_distribution(corpus_filename)

    initialize_embedding(embed, embed.init_type)        # initialize by the specified method
    initialize_network(embed, embed.network_type)

    # determine the position in the tree for every word
    for (w, code) in leaves_of(embed.classification_tree)
        embed.codebook[w] = code
    end

    t1 = time()
    println("Starting sequential training...")
    words_stream = words_of(corpus_filename, subsampling = (embed.subsampling, true, embed.distribution))
run_training!(embed, words_stream; kwargs...)

    t2 = time()
    println("Training complete at $(t2-t1) time")
    embed
end

