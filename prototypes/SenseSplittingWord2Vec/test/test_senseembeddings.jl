using Lumberjack
using FactCheck
using Training
using WordEmbeddings
using Utils

data_dir = joinpath("data") #For local run from testing directory
model_dir = joinpath("models") #For local run from testing directory

test_filename = "text8_tiny"
test_file = joinpath(data_dir, test_filename)
model_file = joinpath(model_dir, test_filename * ".model")





facts("Motions") do
	@fact Training.get_motions([[10.,0 ],[10,0]], 0.0)  --> [[20,0]] "Forces in same direction stack and no breaking"
	@fact Training.get_motions([[10.,0 ],[10,1]], 100.0)  --> [[20,1]] "Forces below break strength different stack"
	@fact Training.get_motions([[10.,0 ],[-5,0]], 0.0)  --> [[10,0],[-5,0]] "Should Break"

	@fact Training.get_motions([[10.,-3 ],[-5,0]], 1.0)  --> [[10,-1.5],[-5,-1.5]] "Should Break, the first dim and share the second"

	@fact Training.get_motions([[10.,0 ],[-5,0],[5,0]], 2.0)  --> [[15,0],[-5,0]] "Should Break and stack in parts"


end



function test_sense_embedding(inputfile)

	embed = WordSenseEmbedding(30, random_inited, huffman_tree, subsampling = 0, iter=2, strength=0.4, force_minibatch_size=100)
	@time train(embed, inputfile)

	save(embed, model_file)
	embed = restore(model_file)
	embed
end

facts("setup correctly") do

	embed = WordSenseEmbedding(30, random_inited, huffman_tree, subsampling = 0, iter=2, strength=0.1)
	append!(embed.vocabulary,["a", "b", "c"])

	initialize_embedding(embed, random_inited)
	
	@fact embed.embedding["a"] |> length --> 1
	@fact embed.embedding["a"][1] |> size --> (30,)
	
	@pending test_sense_embedding(test_file)
end


