using WordEmbeddings
using Training
using Query
using Lumberjack
using Utils

model_dir = "models/ss"
data_dir = "./data/corpora/text8/"
test_filename = "text8"
test_file = joinpath(data_dir, test_filename)

const ndims = 200
const vname =""
base_name  ="$(test_filename)_$(ndims)_$(vname)"
model_file = joinpath(model_dir, base_name)
log_file = joinpath(model_dir, base_name*".log")



function test_word_embedding()
    println("=======================================")
    println("Testing sense splitting word embedding with $base_name")
    println("=======================================")
    
    add_truck(LumberjackTruck(log_file), "filelogger")
    
    embed = FixedWordSenseEmbedding(ndims, random_inited, huffman_tree, subsampling = 0.0,
							  min_count_for_multiple_senses=100, initial_nsenses=20)
    @time train(embed, test_file, 
				end_of_iter_callback=save_callback(model_file),
				end_of_minibatch_callback=sense_counts_callback(model_file)
	)

end

test_word_embedding()
