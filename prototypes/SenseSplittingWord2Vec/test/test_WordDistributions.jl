push!(LOAD_PATH,"../src/")
using FactCheck
using WordDistributions


facts() do 
	
	@fact compute_frequency(Dict("a"=>2,"b"=>2, "c"=>6),10) --> (Dict("a"=>0.2f0,"b"=>0.2f0, "c"=>0.6f0))

	words = IOBuffer("a a b b a a c d a a c c c")
	@fact word_distribution(words, 2) --> (Dict("a"=> 6f0/12, "b"=> 2f0/12, "c"=>4f0/12),12)

end
