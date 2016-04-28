push!(LOAD_PATH,"../src/")
using Word2Vec
using Lumberjack
using FactCheck
using WordEmbeddings

#data_dir = joinpath(Pkg.dir("Word2Vec"), "test", "data")
#model_dir = joinpath(Pkg.dir("Word2Vec"), "test", "models")
data_dir = joinpath("data") #For local run from testing directory
model_dir = joinpath("models") #For local run from testing directory

test_filename = "text8_tiny"
test_file = joinpath(data_dir, test_filename)
model_file = joinpath(model_dir, test_filename * ".model")



function test_sense_embedding(inputfile)

	embed = WordSenseEmbedding(30, Word2Vec.random_inited, Word2Vec.huffman_tree, subsampling = 0, iter=2, strength=0.1)
	@time train(embed, inputfile)

	save(embed, model_file)
	embed = restore(model_file)

end

facts() do

	embed = WordSenseEmbedding(30, Word2Vec.random_inited, Word2Vec.huffman_tree, subsampling = 0, iter=2, strength=0.1)
	append!(embed.vocabulary,["a", "b", "c"])

	Word2Vec.initialize_embedding(embed, Word2Vec.random_inited)
	@fact embed.embedding["a"] |> size --> (30,1)
	

	@pending test_sense_embedding(test_file)
end
