using Utils
using AdaGram

base_name = "more_senses"
param_save_fn =  "../models/adagram/$(base_name).params.jld"
output_fn = "../models/adagram/$(base_name).adagram_model"#"file to save the model (in Julia format)"
@assert !isfile(output_fn)
@param_save param_save_fn begin
	nprocessors = nprocs()
	train_fn  =  "../data/corpora/WikiCorp/tokenised_lowercase_WestburyLab.wikicorp.201004.txt" #"training text data"
	output_fn = output_fn #file to save the model (in Julia format)"
	dict_fn = "../data/corpora/WikiCorp/tokenised_lowercase_WestburyLab.wikicorp.201004.1gram" #"dictionary file with word frequencies"

	window = 10 #"(max) window size" C in the paper
	min_freq  = 20 #"min. frequency of the word"
	remove_top_k = 0 #"remove top K most frequent words"
	dim  = 100 #"dimensionality of representations"
	prototypes = 30 #"number of word prototypes" T in the paper
	alpha = 0.25 #"prior probability of allocating a new prototype"
	d  = 0.0 #"parameter of Pitman-Yor process" D in paper
	subsample = 1e-5 #"subsampling treshold. useful value is 1e-5"
	context_cut  = true #"randomly reduce size of the context"
	epochs = 1 #"number of epochs to train"
	initcount = 1. #"initial weight (count) on first sense for each word"
	stopwords = Set{AbstractString}() #"list of stop words"
	sense_treshold = 1e-10 #"minimal probability of a meaning to contribute into gradients"
	save_treshold = 0.0 #"minimal probability of a meaning to save after training"
end

regex = r"" #"ignore words not matching provided regex"


print("Building dictionary... ")
vm, dict = read_from_file(dict_fn, dim, prototypes, min_freq,remove_top_k, stopwords; regex=regex)
println("Done!")


vm.alpha = alpha
vm.d = d

inplace_train_vectors!(vm, dict, train_fn, window;
                       threshold=subsample, context_cut=context_cut,
					   epochs=epochs, init_count=initcount, sense_treshold=sense_treshold)




save_model(output_fn, vm, dict, save_treshold)




#################

word = "apple"
prior_probs = expected_pi(vm, dict.word2id[word])
for ii in 1:prototypes
    println(prior_probs[ii],"\t",nearest_neighbors(vm, dict, word, ii, 10))
    println()
end

@show disambiguate(vm, dict, "apple", split("fresh tasty breakfast"))
@show disambiguate(vm, dict, "apple", split("new iphone was announced today"))
