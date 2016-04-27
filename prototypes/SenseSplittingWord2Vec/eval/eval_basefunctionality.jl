push!(LOAD_PATH,"../src/")
using Word2Vec
using Base.Test
using Lumberjack

model_dir = "models"
data_dir = "./data/corpora/text8/"
test_filename = "text8"
test_file = joinpath(data_dir, test_filename)

const ndims = 100

base_name  ="$(test_filename)_$ndims"
model_file = joinpath(model_dir, base_name*".model")
log_file = joinpath(model_dir, base_name*".log")



function test_word_embedding()
    println("=======================================")
    println("Testing word embedding with $base_name")
    println("=======================================")
    
    add_truck(LumberjackTruck(log_file), "filelogger")
    
    embed = WordEmbedding(ndims, Word2Vec.random_inited, Word2Vec.huffman_tree; subsampling = 0.0)
    @time train(embed, test_file)

    save(embed, model_file)
    embed = restore(model_file)

    inp = ["king", "queen", "prince", "man", "duke"]
    for w in inp
        info("nearest words to $w")
        info(find_nearest_words(embed, w))
    end
    for c in inp
        target =  "queen-king+"*c
        info(target*" ≈ ")
        info(find_nearest_words(ee, target))
    end
end

test_word_embedding()
