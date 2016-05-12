using Lumberjack
using FactCheck
using Training
using WordEmbeddings
using Utils
using Mocking

data_dir = joinpath("data") #For local run from testing directory
test_file = "./data/text8_miniscule"

remove_truck("console")



facts("Motions") do
#	@fact Training.get_motions([[10.,0 ],[10,0]], 0.0)  --> [[20,0]] "Forces in same direction stack and no breaking"
#	@fact Training.get_motions([[10.,0 ],[10,1]], 100.0)  --> [[20,1]] "Forces below break strength different stack"
#	@fact Training.get_motions([[10.,0 ],[-5,0]], 0.0)  --> [[10,0],[-5,0]] "Should Break"
#
#	@fact Training.get_motions([[10.,-3 ],[-5,0]], 1.0)  --> [[10,-1.5],[-5,-1.5]] "Should Break, the first dim and share the second"
#
#	@fact Training.get_motions([[10.,0 ],[-5,0],[5,0]], 2.0)  --> [[15,0],[-5,0]] "Should Break and stack in parts"

	srand(10) #No actual random chance, in tests
	up_forces   = [round(10.0*rand(Float32, 2)) for ii in 1:10]
	down_forces = [round(-10.0*rand(Float32, 2)) for ii in 1:10]
	forces = shuffle!([up_forces; down_forces])

	@fact Training.get_motions(forces,0.0) |> Set --> Set([sum(up_forces), sum(down_forces)])


end

facts("break and move") do
	function pend(forces, word="a",sense_id=1)
		Dict(word=>Dict(sense_id=>forces))
	end

	embed = WordSenseEmbedding(2, random_inited, huffman_tree, strength=0.0)  
	embed.embedding["a"]=[[0.0,0]]
	@fact Training.break_and_move!(embed, pend([[10., 0]]),"a",1) --> [[10.,0]]
	@fact Training.break_and_move!(embed, pend([[-10., 0]]),"a",1) --> [[0.,0]]
	@fact Training.break_and_move!(embed, pend([[5., 0],[-5,0]]),"a",1) --> [[5.,0],[-5,0]]


	srand(10) #No actual random chance, in tests
	forces   = [ 1.+round(10.0*rand(Float32, 2)) for ii in 1:1000]
	embed.embedding["b"]=[[0.0,0]]
	@fact Training.break_and_move!(embed, pend(forces,"b"),"b",1) |> length  -->  greater_than(0) "should produce some forces after"


	embed.embedding["c"]=[[1.,0],[2.,0]]
	@fact Training.break_and_move!(embed, pend([[50.,-3],[-10,6]],"c",2),"c",2) |> Set --> Set( [[1.,0],[52.,-3],[-8.,6]]) "Don't loose the other word senses"

end



function test_sense_embedding(inputfile)

	embed = WordSenseEmbedding(30, random_inited, huffman_tree, subsampling = 0, iter=2, strength=0.4, force_minibatch_size=100)
	@time train(embed, inputfile)
	embed
end

facts("setup correctly") do

	embed = WordSenseEmbedding(30, random_inited, huffman_tree, subsampling = 0, iter=2, strength=0.1)
	embed.distribution = Dict("a"=>100, "b"=>100, "c"=>100)
	
	Training.initialize_embedding(embed, random_inited)
	
	@fact embed.embedding["a"] |> length --> 1
	@fact embed.embedding["a"][1] |> size --> (30,)
	
	test_sense_embedding(test_file)
end


