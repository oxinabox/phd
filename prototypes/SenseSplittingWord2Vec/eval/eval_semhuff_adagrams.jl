using WordEmbeddings
using Training
using Lumberjack
using Utils
using SemHuff
using SemHuffAdaGram
using AdaGram
using AdaGramCompat

base_name = "semhuff_more_senses"
folder = "../eval/model/adagram"
param_save_fn =  folder*"/$(base_name).params.jld"
output_fn = folder*"/$(base_name).adagram_model.jld"
simsource_fn = "../eval/models/plain/$(base_name).jld"
log_file = base_name*".log"

@assert !isfile(output_fn)
@assert !isfile(simsource_fn)
@param_save param_save_fn begin
	nprocessors = nprocs()
	train_fn  =  "data/corpora/WikiCorp/tokenised_lowercase_WestburyLab.wikicorp.201004.txt" #"training text data"
	output_fn = output_fn #file to save the model (in Julia format)"
	semsimsource_fn = simsource_fn

	window = 10 #"(max) window size" C in the paper
	min_freq  = 20 #"min. frequency of the word"
	remove_top_k = 0 #"remove top K most frequent words"
	dim  = 100 #"dimensionality of representations"
	prototypes = 30 #"number of word prototypes" T	in the paper
	alpha = 0.25 #"prior probability of allocating a new prot	otype"
	d  = 0.0 #"parameter of Pitman-Yor process" D in paper
	subsample = 1e-5 #"subsampling treshold. useful value is 1e-5"
	context_cut  =	 true #"randomly reduce size of the context"
	epochs = 1 #"number of epochs to train"
	initcount = 1. #"initial weight (count) on first sense	for each word"
	stopwords = Set{AbstractString}() #"list of stop words"	
	sense_treshold = 1e-10 #"minimal probability of a meaning to contribu	te into gradients"
	save_treshold = 0.0 #"minimal probability of a mean	ing to save after training"

	semhuff_width = 30 #"When grouping words, consider only the similarities of the closest `semhuff_width` words"
end


function run()
	add_truck(LumberjackTruck(log_file), "filelogger")


	info("################## Training Word Embeddings ###########")
	ee = WordEmbedding(dim, random_inited, huffman_tree,
							  subsampling = 0.0, #Do not subsample, as it breaks my lazy way to calculate freqency
							  min_count=min_freq, iter=1)
	train(embed, test_file,
				end_of_iter_callback=save_callback(model_file,"i")
	)
	
	JLD.save(simsource_fn, "ee", ee)

	

	info("################## Preparing SemHuff ###############")

	
	semtree = semhuff(ee.classification_tree, ee.embeddings, semhuff_width);

	assert(ee.subsampling == 0.0)
	word_freqs =Dict(word=> round(Int64,ee.distribution[word] * ee.corpus_size) 
						for word in keys(ee.distribution))

	ee = nothing #Free the memory

	vm, dict = semhuff_initialize_AdaGram(semtree::Trees.BranchNode,
                                     word_freqs,
                                     dim,
                                     prototypes,
                                     alpha,
                                     d)

	info("################### Training Adagrams  ####################")

	inplace_train_vectors!(vm, dict, train_fn, window;
	                       threshold=subsample, context_cut=context_cut,
						   epochs=epochs, init_count=initcount, 
						   sense_treshold=sense_treshold)
	am = AdaGramModel(vm, dict)
	JLD.save(output_fn, "am", am)

	info("Done")
end

run()
