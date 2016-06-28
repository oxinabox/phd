using Lumberjack
using Base.Test
using Training
using WordEmbeddings
using Utils

test_file = "./data/text8_miniscule"
remove_truck("console")


@testset "setup correctly" begin

	embed = FixedWordSenseEmbedding(30, random_inited, huffman_tree, subsampling = 0, iter=2, 
									min_count_for_multiple_senses=1_000_000_000) #Make never do multiple senses for this test
	Training.setup!(embed, test_file)
	original_vec_a1=copy(embed.embedding["a"][1])
	
	@time resume_training!(embed, test_file)
	@test embed.trained_times["a"][1] > 0
	@test embed.embedding["a"][1] != original_vec_a1 #Should have changed

end


