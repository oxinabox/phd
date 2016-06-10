using Query
using Base.Test
using WordEmbeddings

embed = WordEmbedding(2, random_inited, huffman_tree)
embed.embedding["a"] = [1., 1.]
embed.embedding["b"] = [11., 11.]
embed.embedding["c"] = [5., 2.]
embed.embedding["d"] = [15., 12.]
embed.embedding["e"] = [4., 2.]

@testset "Nearest Words Math" begin
	@test [w for (w,v) in find_nearest_words(embed,"c", nwords=2)] == ["e", "d"]

	@test find_nearest_words(embed, "b-a+c", nwords=1)[1][1] == "d" #"pattern given in http://www.aclweb.org/anthology/N13-1090"

	@test_throws KeyError find_nearest_words(embed,"z") 
end

sembed = FixedWordSenseEmbedding(2, random_inited, huffman_tree)
sembed.embedding["a"] = [[1., 1.],[100,101.]]
sembed.embedding["b"] = [[11., 11.],[100,102.]]
sembed.embedding["c"] = [[5., 2.],[100,103.]]
sembed.embedding["d"] = [[15., 12.],[100,104.]]
sembed.embedding["e"] = [[4., 2.],[100,105.]]


@testset "Nearest Words Sense Math" begin
	@test [w for (w,v) in find_nearest_words(sembed,"c",1, nwords=2)] == [("e",1), ("d",1)]
end




@testset "Context Prob" begin
	@test logprob_of_context(embed,["z"],"a"; skip_oov=true) == 0.0

	@test_throws KeyError logprob_of_context(embed,["z"],"a")
end

