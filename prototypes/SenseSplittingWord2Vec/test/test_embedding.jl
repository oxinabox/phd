push!(LOAD_PATH,"../src/")
using Word2Vec
using Lumberjack
using FactCheck

#data_dir = joinpath(Pkg.dir("Word2Vec"), "test", "data")
#model_dir = joinpath(Pkg.dir("Word2Vec"), "test", "models")
data_dir = joinpath("data") #For local run from testing directory
model_dir = joinpath("models") #For local run from testing directory

test_filename = "text8_tiny"
test_file = joinpath(data_dir, test_filename)
model_file = joinpath(model_dir, test_filename * ".model")



type CheckAlphaDecreasingTruck <:TimberTruck
	prev_α::Float64
	_mode
end


function Lumberjack.log(t::CheckAlphaDecreasingTruck, args::Dict)
	if haskey(args,:α)
		α = args[:α]
		@fact α --> less_than_or_equal(t.prev_α) "Learning Rate (α) must not decrease"
		t.prev_α=α
	end
end



function test_word_embedding(inputfile)
	embed = WordEmbedding(30, Word2Vec.random_inited, Word2Vec.huffman_tree, subsampling = 0, iter=2)
	add_truck(CheckAlphaDecreasingTruck(Inf,"info"), "Testing Truck")

	@time train(embed, inputfile)

	save(embed, model_file)
	embed = restore(model_file)

	inp = ["king", "queen", "prince"]
	for w in inp
		println("nearest words to $w")
		println(find_nearest_words(embed, w))
	end
	println("king + prince - queen ≈?")
	println(find_nearest_words(embed, ["king", "prince"], ["queen"]))
end

facts("Some Facts about Word Embedding Training") do
	test_word_embedding(test_file)
end
