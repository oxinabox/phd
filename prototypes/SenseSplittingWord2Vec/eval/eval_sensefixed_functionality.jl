using WordEmbeddings
using Training
using Query
using Lumberjack
using Utils

model_dir = "models/ss"
test_filename = "tokenised_lowercase_WestburyLab.wikicorp.201004"
#test_filename = "text8"
#test_file = "./data/corpora/text8/text8"
test_file = "./data/corpora/WikiCorp/"*test_filename*".txt"
const ndims = 100
const vname =""
base_name  ="$(test_filename)_$(ndims)_$(vname)"
model_file = joinpath(model_dir, base_name)
log_file = joinpath(model_dir, base_name*".log")



function test_word_embedding()
    println("=======================================")
    println("Testing sensefixed word embedding with $base_name")
    println("=======================================")
    
    add_truck(LumberjackTruck(log_file), "filelogger")
    
    embed = FixedWordSenseEmbedding(ndims, random_inited, huffman_tree, 
							  subsampling = 10.0^-5.0,
							  min_count_for_multiple_senses=20_000, 
							  initial_nsenses=20,
							  force_minibatch_size=10_000_000,
							  min_count=250, iter=1)
		 #wikicorp counts: 
		 #				   20_000 means most common 5000 words, 
		 #				     8000            ->  mc  10_000 words,
		 #					 3000			 ->  mc  20_000 words
		 #				     1600            ->  mc  30_000 words, 
		 #					  250            ->  mc 100_000 words
		 #						5			 ->  mc 1.5×10⁶ words
	@time train(embed, test_file,
                end_of_iter_callback=save_callback(model_file,"m"),
				end_of_minibatch_callback=save_callback(model_file,"m"),
	)

end

test_word_embedding()
