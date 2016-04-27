module WordEmbeddings
using Trees
i
export RandomInited, HuffmanTree, NaiveSoftmax, random_inited, naive_softmax, huffman_tree, WordEmbedding, keep_word_vectors_only!


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


type WordEmbedding
    vocabulary::Array{AbstractString}
    embedding::Dict{AbstractString, Vector{Float64}}
    classification_tree::TreeNode
    distribution::Dict{AbstractString, Float64}
    codebook::Dict{AbstractString, Vector{Int64}}

    init_type::InitializatioinMethod
    network_type::NetworkType
    dimension::Int64
    lsize::Int64    # left window size in training
    rsize::Int64    # right window size
    trained_times::Dict{AbstractString,Int64}
    trained_count::Int64
    corpus_size::Int64
    subsampling::Float64
    init_learning_rate::Float64
    iter::Int64
    min_count::Int64
end

function WordEmbedding(dim::Int64, init_type::InitializatioinMethod, network_type::NetworkType; lsize=5, rsize=5, subsampling=1e-5, init_learning_rate=0.025, iter=5, min_count=5)
    if dim <= 0 || lsize <= 0 || rsize <= 0
        throw(ArgumentError("dimension should be a positive integer"))
    end
    WordEmbedding(AbstractString[], 
                    Dict{AbstractString,Array{Float64}}(),
                    nullnode,
                    Dict{AbstractString,Array{Float64}}(),
                    Dict{AbstractString,Vector{Int64}}(),
                    init_type, network_type,
                    dim,
                    lsize, rsize,
                    Dict{AbstractString,Int64}(),
                    0, 0,
                    subsampling, init_learning_rate, iter, min_count)
end

function Base.show(io::IO, x::WordEmbedding)
    println(io, "Word embedding(dimension = $(x.dimension))"*
			"of $(length(x.vocabulary)) words, trained on $(x.trained_count) words")
    nothing
end

# strip embedding and retain only word vectors
function keep_word_vectors_only!(embed::WordEmbedding)
    embed.vocabulary = AbstractString[]
    embed.classification_tree = nullnode
    embed.distribution = Dict{AbstractString,Array{Float64}}()
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


end #module
