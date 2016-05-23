using WordEmbeddings
using Training
using Query
using Lumberjack
using Utils

model_dir = "models/ss"
data_dir = "./data/corpora/text8/"
test_filename = "text8"
test_file = joinpath(data_dir, test_filename)

const ndims = 50

base_name  ="$(test_filename)_$ndims"
model_file = joinpath(model_dir, base_name*".model")
log_file = joinpath(model_dir, base_name*".log")



function test_word_embedding()
    println("=======================================")
    println("Testing sense splitting word embedding with $base_name")
    println("=======================================")
    
    add_truck(LumberjackTruck(log_file), "filelogger")
    
    embed = FixedWordSenseEmbedding(ndims, random_inited, huffman_tree, subsampling = 0.0,
							  force_minibatch_size=50_000)
    @time train(embed, test_file, 
				end_of_iter_callback=save_callback(model_file),
				end_of_minibatch_callback=sense_counts_callback(model_file)
	)

end

test_word_embedding()
