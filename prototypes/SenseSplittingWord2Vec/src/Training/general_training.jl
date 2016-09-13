#*===============================================================================
#* Calculate Word Embedding
#*Train a corpus
#*Step 1: Find Word Distribution
#*Step 2: Calculate Word Embedding
#*===============================================================================#

############ Callbacks
function save_callback(basefilename::String, prefix="i")
	function scallbck(arg)
		iter, embed = arg
		
		save(embed, "$(basefilename)_$(prefix)$(iter).model")
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


function initialize_network!(embed::GenWordEmbedding, network_type::SemHuffTree)
    embed.classification_tree = transform_tree(semtree, 
                            leaf_transform = word->word,
						    internal_transform = dummy -> LinearClassifier(2,embed.dim))
    
    embed.codebook = Dict(leaves_of(classification_tree))
    embed
end


initialize_network(embed::GenWordEmbedding) = initialize_network(embed, embed.network_type)

function initialize_network(embed::GenWordEmbedding, huffman::HuffmanTree)
	embed.classification_tree, embed.codebook = initialize_network(embed.distribution, embed.dimension, huffman)
    embed
end


function initialize_network{S<:String, N<:Number}(distribution::Associative{S,N}, embedding_dim::Number, ::HuffmanTree)
    heap = PriorityQueue()
    for (word, freq) in distribution
        node = BranchNode([], word)    # the data field of leaf node is its corresponding word.
        enqueue!(heap, node, freq)
    end
    while length(heap) > 1
        (node1, freq1) = Base.Collections.peek(heap)
        dequeue!(heap)
        (node2, freq2) = Base.Collections.peek(heap)
        dequeue!(heap)
        newnode = BranchNode([node1, node2], LinearClassifier(2, embedding_dim))
								#the data field of internal node is the classifier
        enqueue!(heap, newnode, freq1 + freq2)
    end
    classification_tree = dequeue!(heap)

	codebook = Dict(leaves_of(classification_tree))
	classification_tree, codebook
end

initialize_embedding(embed::GenWordEmbedding) = initialize_embedding(embed, embed.init_type)

function resume_training!(embed::GenWordEmbedding, corpus_filename::String, initial_trained_count = 0; kwargs...)
    t1 = time()
    println("Starting sequential training...")
    words_stream = words_of(corpus_filename, subsampling = (embed.subsampling, true, embed.distribution))

	run_training!(embed, words_stream; initial_trained_count=initial_trained_count, kwargs...)

    t2 = time()
    println("Training complete at $(t2-t1) time")
    embed
end

function setup!(embed::GenWordEmbedding, corpus_filename::String)

    embed.distribution, full_corpus_size = word_distribution(corpus_filename, embed.min_count)
	
	embed.corpus_size = round(subsampled_wordcount(embed.subsampling, embed.distribution, full_corpus_size))
	@assert(embed.corpus_size>0, "embed.corpus_size = $(embed.corpus_size) <= 0")
	@assert(embed.corpus_size<=full_corpus_size)

    initialize_embedding(embed)        # initialize by the specified method
    initialize_network(embed)
	embed
end

function train(embed::GenWordEmbedding, corpus_filename::String; kwargs...)
	setup!(embed, corpus_filename)
	resume_training!(embed, corpus_filename; kwargs...)
end

