using Base.Test
using WordDistributions


@testset "WordDistributions" begin
	
	@test compute_frequency(Dict("a"=>2,"b"=>2, "c"=>6),10) == (Dict("a"=>0.2f0,"b"=>0.2f0, "c"=>0.6f0))
	words = IOBuffer("a a b b a a c d a a c c c")
	@test word_distribution(words, 2) == (Dict("a"=> 6f0/12, "b"=> 2f0/12, "c"=>4f0/12),12)


	@test subsampled_wordcount(.00005, Dict("a"=> 6f0/12, "b"=> 2f0/12, "c"=>4f0/12), 100) < 100
	@test subsampled_wordcount(.00005, Dict("a"=> 6f0/12, "b"=> 2f0/12, "c"=>4f0/12), 1000) > 1

	@test subsampled_wordcount(.0, Dict("a"=> 6f0/12, "b"=> 2f0/12, "c"=>4f0/12), 1000) ≈ 1000
end
