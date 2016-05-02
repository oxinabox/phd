#*===============================================================================
#* Calculate Word Embedding
#*Train a corpus
#*Step 1: Find Word Distribution
#*Step 2: Calculate Word Embedding
#*===============================================================================#


"Runs all the training, handles adjusting learning rate, repeating through loops etc."
function run_training!(embed::GenWordEmbedding, words_stream::WordStream, strip::Bool=false)
    tic()
	middle = embed.lsize + 1
    input_gradient = zeros(Float32, embed.dimension)
    α = embed.init_learning_rate
    trained_count = 0
    trained_times = Dict{AbstractString, Int64}()

    for current_iter in 1:embed.iter
	debug("Iter $current_iter of $(embed.iter)")
        for (current_iter_prog,window) in enumerate_progress(sliding_window(words_stream, lsize=embed.lsize, rsize=embed.rsize))
            trained_word = window[middle]
            trained_times[trained_word] = get(trained_times, trained_word, 0) + 1
            trained_count += 1

            if trained_count % 10000 == 0
                progress = (current_iter-1 + current_iter_prog)/ embed.iter
                info("trained on $trained_count words"; progress=progress, α=α)
                α = embed.init_learning_rate * (1 - progress)
                if α < embed.init_learning_rate * 0.0001
                    α = embed.init_learning_rate * 0.0001
                end
            end
			train_window!(embed,window,middle,input_gradient,α)
        end
    end

    embed.trained_count = trained_count
    embed.trained_times = trained_times

    overall_time = toq()
    debug("Finished training. Trained on $(embed.trained_count) words in $(overall_time) seconds.")

    # strip to remove unnecessary members and make serialization faster
    strip && keep_word_vectors_only!(embed)
    embed
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
        newnode = BranchNode([node1, node2], LinearClassifier(2, embed.dimension)) # the data field of internal node is the classifier
        enqueue!(heap, newnode, freq1 + freq2)
    end
    embed.classification_tree = dequeue!(heap)
    embed
end


function train(embed::GenWordEmbedding, corpus_filename::AbstractString)
    embed.distribution = word_distribution(corpus_filename)
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
    run_training!(embed, words_stream, false)

    t2 = time()
    println("Training complete at $(t2-t1) time")
    embed
end



