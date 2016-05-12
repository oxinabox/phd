module WordEmbeddings
using Trees

export RandomInited, HuffmanTree, NaiveSoftmax, random_inited, naive_softmax, huffman_tree, GenWordEmbedding, keep_word_vectors_only!, WordEmbedding, WordSenseEmbedding


# The types defined below are used for specifying the options of the word embedding training
abstract Option

abstract InitializatioinMethod <: Option
type RandomInited <: InitializatioinMethod 
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
    embedding::Dict{AbstractString, Vector{Float32}}
    classification_tree::TreeNode
    distribution::Dict{AbstractString, Float32}
    codebook::Dict{AbstractString, Vector{Int64}}

    init_type::InitializatioinMethod
    network_type::NetworkType
    dimension::Int64
    lsize::Int64    # left window size in training
    rsize::Int64    # right window size
    trained_times::Dict{AbstractString,Int64}
    corpus_size::Int64
    subsampling::Float32
    init_learning_rate::Float32
    iter::Int64
    min_count::Int64
end

function WordEmbedding(dim::Int64, init_type::InitializatioinMethod, network_type::NetworkType; lsize=5, rsize=5, subsampling=1e-5, init_learning_rate=0.025, iter=5, min_count=5)
    if dim <= 0 || lsize <= 0 || rsize <= 0
        throw(ArgumentError("dimension should be a positive integer"))
    end
    WordEmbedding(
                    Dict{AbstractString,Array{Float32}}(),
                    nullnode,
                    Dict{AbstractString,Array{Float32}}(),
                    Dict{AbstractString,Vector{Int64}}(),
                    init_type, network_type,
                    dim,
                    lsize, rsize,
                    Dict{AbstractString,Int64}(),
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
    embed.distribution = AbstractString[]
    embed.classification_tree = nullnode
    embed.distribution = Dict{AbstractString,Array{Float32}}()
    embed.codebook = Dict{AbstractString,Vector{Int64}}()
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


type WordSenseEmbedding<:GenWordEmbedding
    embedding::Dict{AbstractString, Vector{Vector{Float32}}} #[Word][sense_id]=sense embedding vector
    classification_tree::TreeNode
    distribution::Dict{AbstractString, Float32}
    codebook::Dict{AbstractString, Vector{Int64}}

	strength::Float32
	
    force_minibatch_size::Int64

	init_type::InitializatioinMethod
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

function WordSenseEmbedding(dim::Int64, init_type::InitializatioinMethod, network_type::NetworkType;
							lsize=5, rsize=5, subsampling=1e-5, init_learning_rate=0.025, iter=5,
							min_count=5, force_minibatch_size=50_000, strength=0.8)
    if dim <= 0 || lsize <= 0 || rsize <= 0
        throw(ArgumentError("dimension should be a positive integer"))
    end
	if force_minibatch_size<min_count
        throw(ArgumentError("min_count must be at least equal to force_minibatch_size, so that rare words are not resolved less than once per interation"))
    end
	
    WordSenseEmbedding(
                    Dict{AbstractString,Vector{Vector{Float32}}}(), #embedding
                    nullnode, #classification tree
                    Dict{AbstractString,Array{Float32}}(), #distribution
                    Dict{AbstractString,Vector{Int64}}(), #codebook
					strength,
					force_minibatch_size,
                    init_type, network_type,
                    dim,
                    lsize, rsize,
                    0, 
                    subsampling, init_learning_rate, iter, min_count)
end


end #module
