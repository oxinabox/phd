using WordEmbeddings
using Training
using Query
using Lumberjack
using Utils

model_dir = "models"
data_dir = "./data/corpora/text8/"
test_filename = "text8"
test_file = joinpath(data_dir, test_filename)

const ndims = 30

base_name  ="$(test_filename)_$ndims"
model_file = joinpath(model_dir, base_name*".model")
log_file = joinpath(model_dir, base_name*".log")



function test_word_embedding()
    println("=======================================")
    println("Testing word embedding with $base_name")
    println("=======================================")
    
    add_truck(LumberjackTruck(log_file), "filelogger")
    
    embed = WordEmbedding(ndims, random_inited, huffman_tree, subsampling = 0.0)
    @time train(embed, test_file, end_of_iter_callback=save_callback(base_name))


    inp = ["king", "queen", "prince", "man", "duke"]
    for w in inp
        info("nearest words to $w")
        info(find_nearest_words(embed, w))
    end
    for c in inp
        target =  "queen-king+"*c
        info(target*" ≈ ")
        info(find_nearest_words(embed, target))
    end
end

test_word_embedding()
