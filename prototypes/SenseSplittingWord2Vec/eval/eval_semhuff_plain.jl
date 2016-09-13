using WordEmbeddings
using Training
using Lumberjack
using Utils
using SemHuff
using SwiftObjectStores


base_name = "semhuff_plain"
detailed_name = "tokenised_lowercase_WestburyLab.wikicorp.201004_100_nosubsample"
folder = "models/adagram"
param_save_fn =  folder*"/$(base_name).params.jld"
simsource_fn = "models/plain/"*detailed_name*".jld"
log_file = base_name*".log"

#@assert !isfile(simsource_fn)
@param_save param_save_fn begin
	train_fn  =  "data/corpora/WikiCorp/tokenised_lowercase_WestburyLab.wikicorp.201004.txt" #"training text data"
	semsimsource_fn = simsource_fn
	semhuff_tree_fn = ("sensemodels", "semhuff_trees/"*detailed_name*".semhuff_tree.jld")
	output_fn = ("sensemodels", "plain/"*detailed_name*".semhuff.jld")


	window = 10 #"(max) window size" C in the paper
	min_freq  = 20 #"min. frequency of the word"
	dim  = 100 #"dimensionality of representations"
	subsample = 1e-5 #"subsampling treshold. useful value is 1e-5"
end


function run()
	add_truck(LumberjackTruck(log_file), "filelogger")

	serv = SwiftService()

	#info("################## Training Word Embeddings ###########")
	#ee = WordEmbedding(dim, random_inited, huffman_tree,
	#						  subsampling = 0.0, #Do not subsample, as it breaks my lazy way to calculate freqency
	#						  min_count=min_freq, iter=1)
	#train(ee, train_fn)
	
	#JLD.save(simsource_fn, "ee", ee)

	#ee = JLD.load(simsource_fn, "ee")
	


	info("################## Preparing SemHuff ###############")

	
	#semtree = semhuff(ee.classification_tree, ee.embedding, semhuff_width);
	#JLD.save(semhuff_tree_fn, "semtree", semtree)
	
	semtree = get_jld(serv, semhuff_tree_fn..., "semtree")

#	ee = nothing #Free the memory

	

	info("########################### Training Model ######")
	embed = WordEmbedding(dim, random_inited, huffman_tree, #SemHuffTree(semtree), 
					  subsampling = subsample,
					  min_count=min_freq)

	train(embed, train_fn)
	
	info("##Uploading##")
	put_jld(serv, output_fn..., ee=embed)


	info("Done")
end

run()
