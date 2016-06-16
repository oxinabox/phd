using WordEmbeddings
using Training
using Query
using Lumberjack
using Utils

model_dir = "models/ss"
test_filename = "WestburyLab.wikicorp.201004"
test_file = "data/corpora/WikiCorp/tokenised_WestburyLab.wikicorp.201004.txt"


const ndims = 300
const vname ="v2"
base_name  ="$(test_filename)_$(ndims)_$(vname)"
model_file = joinpath(model_dir, base_name)
log_file = joinpath(model_dir, base_name*".log")



function test_word_embedding()
    println("=======================================")
    println("Testing sense splitting word embedding with $base_name")
    println("=======================================")
    
    add_truck(LumberjackTruck(log_file), "filelogger")
    
    embed = FixedWordSenseEmbedding(ndims, random_inited, huffman_tree, 
							  subsampling = 10.0^-5.0,
							  min_count_for_multiple_senses=20_000, initial_nsenses=20,
							  min_count=1000, iter=1)
    @time train(embed, test_file, 
				end_of_iter_callback=save_callback(model_file),
	)

end

test_word_embedding()
