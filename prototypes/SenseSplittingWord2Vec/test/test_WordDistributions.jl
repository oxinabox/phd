push!(LOAD_PATH,"../src/")
using FactCheck
using WordDistributions


facts() do 
	
	@fact compute_frequency(Dict("a"=>2,"b"=>2, "c"=>6),10) --> (Dict("a"=>0.2,"b"=>0.2, "c"=>0.6))

	words = IOBuffer("a a b b a a c d a a c c c")
	@fact word_distribution(words, 2) --> Dict("a"=> 6/12, "b"=> 2/12, "c"=>4/12)
end
