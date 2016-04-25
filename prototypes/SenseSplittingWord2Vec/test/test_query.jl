push!(LOAD_PATH,"../src/")
using Word2Vec
using FactCheck


embed = WordEmbedding(30, Word2Vec.random_inited, Word2Vec.huffman_tree)
embed.embedding["a"] = [1., 1.]
embed.embedding["b"] = [11., 11.]
embed.embedding["c"] = [5., 2.]
embed.embedding["d"] = [15., 12.]
embed.embedding["e"] = [4., 2.]

facts() do
	@fact [w for (w,v) in find_nearest_words(embed,"c", nwords=2)] --> ["e", "d"]

	@fact find_nearest_words(embed, "a - b+c", nwords=1)[1][1] --> "d"

	@fact_throws find_nearest_words(embed,"z") KeyError
end
