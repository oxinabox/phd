module Word2Vec

using Base.Collections      # for priority queue
using Base.Cartesian        # for @nexprs
using Distances
using Blocks
using Compat


export LinearClassifier, train_one, WordEmbedding, train, accuracy
export save, restore
export find_nearest_words

include("utils.jl")
include("tree.jl")
include("word_stream.jl")
include("softmax_classifier.jl")
include("train.jl")
include("query.jl")

end # module
