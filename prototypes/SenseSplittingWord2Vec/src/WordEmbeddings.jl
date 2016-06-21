module WordEmbeddings
using Trees
using DataStructures

export RandomInited, HuffmanTree, NaiveSoftmax, random_inited, naive_softmax, huffman_tree, GenWordEmbedding, keep_word_vectors_only!, WordEmbedding, WordSenseEmbedding, FixedWordSenseEmbedding, SplittingWordSenseEmbedding


# The types defined below are used for specifying the options of the word embedding training
abstract Option

abstract InitializationMethod <: Option
type RandomInited <: InitializationMethod 
    # Initialize the embedding randomly
end
const random_inited = RandomInited()

abstract NetworkType <: Option
type NaiveSoftmax <: NetworkType
    # |V| outputs softmax
end
type HuffmanTree <: NetworkType
    # Predicate step by step on the huffman tree
end
const naive_softmax = NaiveSoftmax()
const huffman_tree = HuffmanTree()


abstract GenWordEmbedding

##################### (plain)  Word Embeddings ##################################

type WordEmbedding<:GenWordEmbedding
    embedding::Dict{String, Vector{Float32}}
    classification_tree::TreeNode
    distribution::Dict{String, Float32}
    codebook::Dict{String, Vector{Int64}}

    init_type::InitializationMethod
    network_type::NetworkType
    dimension::Int64
    lsize::Int64    # left window size in training
    rsize::Int64    # right window size
    trained_times::Dict{String,Int64}
    corpus_size::Int64
    subsampling::Float32
    init_learning_rate::Float32
    iter::Int64
    min_count::Int64
end

function WordEmbedding(dim::Int64, init_type::InitializationMethod, network_type::NetworkType; lsize=5, rsize=5, subsampling=1e-5, init_learning_rate=0.025, iter=5, min_count=5)
    if dim <= 0 || lsize <= 0 || rsize <= 0
        throw(ArgumentError("dimension should be a positive integer"))
    end
    WordEmbedding(
                    Dict{String,Array{Float32}}(),
                    nullnode,
                    Dict{String,Array{Float32}}(),
                    Dict{String,Vector{Int64}}(),
                    init_type, network_type,
                    dim,
                    lsize, rsize,
                    Dict{String,Int64}(),
                    0, #corpus size
                    subsampling, init_learning_rate, iter, min_count)
end

function Base.show(io::IO, x::WordEmbedding)
    println(io, "Word embedding(dimension = $(x.dimension))"*
			"of $(length(x.distribution)) words, trained on $(x.trained_count) words")
    nothing
end

# strip embedding and retain only word vectors
function keep_word_vectors_only!(embed::WordEmbedding)
    embed.classification_tree = nullnode
    embed.distribution = Dict{String,Array{Float32}}()
    embed.codebook = Dict{String,Vector{Int64}}()
    embed
end

# print the code book for debug
function _print_codebook(embed::WordEmbedding, N=10)
    for (word,code) in embed.codebook
        println("$word => $code")
        N -= 1
        (N > 0) || break
    end
    nothing
end

##################### Word Sense  Embeddings ##################################
abstract WordSenseEmbedding<:GenWordEmbedding

type SplittingWordSenseEmbedding<:WordSenseEmbedding
    embedding::Dict{String, Vector{Vector{Float32}}} #[Word][sense_id]=sense embedding vector
    classification_tree::TreeNode
    distribution::Dict{String, Float32}
    codebook::Dict{String, Vector{Int64}}


	strength::Float32
	nsplitaxes::Int64	
    force_minibatch_size::Int64

	init_type::InitializationMethod
    network_type::NetworkType
    dimension::Int64
    lsize::Int64    # left window size in training
    rsize::Int64    # right window size
    corpus_size::Int64
    subsampling::Float32
    init_learning_rate::Float32
    iter::Int64
    min_count::Int64
end

function SplittingWordSenseEmbedding(dim::Int64, init_type::InitializationMethod, network_type::NetworkType;
							lsize=5, rsize=5, subsampling=1e-5, init_learning_rate=0.025, iter=5,
							min_count=5, force_minibatch_size=50_000, strength=0.8, nsplitaxes=-1)
    if dim <= 0 || lsize <= 0 || rsize <= 0
        throw(ArgumentError("dimension should be a positive integer"))
    end
	if force_minibatch_size<min_count
        throw(ArgumentError("min_count must be at least equal to force_minibatch_size, so that rare words are not resolved less than once per interation"))
    end
	if nsplitaxes<0 #Not set
		nsplitaxes = dim #Default to split everywhere
	end
	
    SplittingWordSenseEmbedding(
                    Dict{String,Vector{Vector{Float32}}}(), #embedding
                    nullnode, #classification tree
                    Dict{String,Array{Float32}}(), #distribution
                    Dict{String,Vector{Int64}}(), #codebook
					strength,
					nsplitaxes,
					force_minibatch_size,
                    init_type, network_type,
                    dim,
                    lsize, rsize,
                    0, 
                    subsampling, init_learning_rate, iter, min_count)
end




type FixedWordSenseEmbedding<:WordSenseEmbedding
    embedding::Dict{String, Vector{Vector{Float32}}} #[Word][sense_id]=sense embedding vector
    classification_tree::TreeNode
    distribution::Dict{String, Float32}
    codebook::Dict{String, Vector{Int64}}
	trained_times::Associative #TODO: give this is a clear type
    force_minibatch_size::Int64

	init_type::InitializationMethod
    network_type::NetworkType
    dimension::Int64
    lsize::Int64    # left window size in training
    rsize::Int64    # right window size
    corpus_size::Int64
    subsampling::Float32
    init_learning_rate::Float32
    iter::Int64
    min_count::Int64
	min_count_for_multiple_senses::Int64
	initial_nsenses::Int64
end

function FixedWordSenseEmbedding(dim::Int64, init_type::InitializationMethod, network_type::NetworkType;
							lsize=5, rsize=5, subsampling=1e-5, init_learning_rate=0.025, iter=5,
							force_minibatch_size=50_000,
							min_count=5, min_count_for_multiple_senses=100, initial_nsenses=20
							)
    if dim <= 0 || lsize <= 0 || rsize <= 0
        throw(ArgumentError("dimension should be a positive integer"))
    end
	
    FixedWordSenseEmbedding(
                    Dict{String,Vector{Vector{Float32}}}(), #embedding
                    nullnode, #classification tree
                    Dict{String,Array{Float32}}(), #distribution
                    Dict{String,Vector{Int64}}(), #codebook
					DefaultDict(String,DefaultDict{Int64,Int64},()->DefaultDict(Int64,Int64,()->0)), #Trained times
					force_minibatch_size,
                    init_type, network_type,
                    dim,
                    lsize, rsize,
                    0, 
                    subsampling, init_learning_rate, iter, min_count,
					min_count_for_multiple_senses, initial_nsenses
					)
end



end #Module
