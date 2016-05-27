using Query
using Base.Test
using WordEmbeddings

embed = WordEmbedding(30, random_inited, huffman_tree)
embed.embedding["a"] = [1., 1.]
embed.embedding["b"] = [11., 11.]
embed.embedding["c"] = [5., 2.]
embed.embedding["d"] = [15., 12.]
embed.embedding["e"] = [4., 2.]

@testset "Nearest Words" begin
	@test [w for (w,v) in find_nearest_words(embed,"c", nwords=2)] == ["e", "d"]

	@test find_nearest_words(embed, "b-a+c", nwords=1)[1][1] == "d" #"pattern given in http://www.aclweb.org/anthology/N13-1090"

	@test_throws KeyError find_nearest_words(embed,"z") 
end

@testset "Context Prob" begin
	@test logprob_of_context(embed,["z"],"a"; skip_oov=true) == 0.0

	@test_throws KeyError logprob_of_context(embed,["z"],"a")
end

