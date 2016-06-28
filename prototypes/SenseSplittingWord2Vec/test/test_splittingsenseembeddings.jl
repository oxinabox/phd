using Lumberjack
using Base.Test
using Training
using WordEmbeddings
using Utils

data_dir = joinpath("data") #For local run from testing directory
test_file = "./data/text8_miniscule"
remove_truck("console")

@testset "Motions" begin
	@test Training.get_motions([[10.,0 ],[10,0]], 0.0)  == [[20,0]]  #Forces in same direction stack and no breaking
	@test Training.get_motions([[10.,0 ],[10,1]], 100.0)  == [[20,1]]  #Forces below break strength different stack
	@test Training.get_motions([[10.,0 ],[-5,0]], 0.0)  ≅ [[10,0],[-5,0]]  #Should Break

	@test Training.get_motions([[10.,-3 ],[-5,0]], 1.0) ≅ [[10,-3],[-5,0]] 

	@test Training.get_motions([[10.,0 ],[-5,0],[5,0]], 2.0) ≅ [[15,0],[-5,0]]  #Should Break and stack in parts

	srand(10) #No actual random chance, in tests
	up_forces   = [[round(10.0*rand(Float32)),0] for ii in 1:10]
	down_forces = [[round(-10.0*rand(Float32)),0] for ii in 1:10]
	forces = shuffle!([up_forces; down_forces])

	@test Training.get_motions(forces,0.0) ≅ [sum(up_forces), sum(down_forces)]
	
	forces_2  = [round(10.0*rand(Float32,30)) for ii in 1:30]
	@test Training.get_motions(forces_2, Inf) ==[sum(forces_2)] #Infinite strength means only one result


end


@testset "break and move" begin
	function sdict(forces, sense_id=1)
		Dict(sense_id=>forces)
	end

	zero_start = [[0.0f0,0.0]]	
	#@test Training.break_and_move!(zero_start,  sdict([[10f0, 0]]), 0.) ==	[[10f0,0]]
	#@test Training.break_and_move!(zero_start,  sdict([[-10f0, 0]]), 0.) == [[0f0,0]]
	#@test Training.break_and_move!(zero_start,  sdict([[5f0, 0],[-5,0]]), 0.) == [[5f0,0],[-5f0,0]]


	#srand(10) #No actual random chance, in tests
	#forces   = [ 1.+round(10.0*rand(Float32, 2)) for ii in 1:1000]
	#@test Training.break_and_move!(zero_start, sdict(forces),0.0) |> length  > 0   #should produce some forces after


	#@test Training.break_and_move!([[1f0,0],[2f0,0]],
#								sdict([[50f0,-3],[-10f0,6]],2), 0.0)  ≅ [[1f0,0],[52f0,-3],[-8f0,6]] # "Don't loose the other word senses"

end


function test_sense_embedding(inputfile)

	embed = SplittingWordSenseEmbedding(30, random_inited, huffman_tree, subsampling = 0, iter=2, strength=0.4, force_minibatch_size=100)
	@time train(embed, inputfile)
	embed
end

@testset "setup correctly" begin

	embed = SplittingWordSenseEmbedding(30, random_inited, huffman_tree, subsampling = 0, iter=2, strength=0.1)
	embed.distribution = Dict("a"=>100, "b"=>100, "c"=>100)
	
	Training.initialize_embedding(embed, random_inited)
	
	@test embed.embedding["a"] |> length == 1
	@test embed.embedding["a"][1] |> size == (30,)
	
	test_sense_embedding(test_file)
end


