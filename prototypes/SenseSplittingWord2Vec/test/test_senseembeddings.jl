using Lumberjack
using FactCheck
using Word2Vec.Training
using WordEmbeddings
using Utils

data_dir = joinpath("data") #For local run from testing directory
model_dir = joinpath("models") #For local run from testing directory

test_filename = "text8_tiny"
test_file = joinpath(data_dir, test_filename)
model_file = joinpath(model_dir, test_filename * ".model")



function test_sense_embedding(inputfile)

	embed = WordSenseEmbedding(30, random_inited, huffman_tree, subsampling = 0, iter=2, strength=0.1)
	@time train(embed, inputfile)

	save(embed, model_file)
	embed = restore(model_file)

end

facts() do

	embed = WordSenseEmbedding(30, random_inited, huffman_tree, subsampling = 0, iter=2, strength=0.1)
	append!(embed.vocabulary,["a", "b", "c"])

	initialize_embedding(embed, random_inited)
	
	@fact embed.embedding["a"] |> length --> 1
	@fact embed.embedding["a"][1] |> size --> (30,)
	

	@pending test_sense_embedding(test_file)
end


