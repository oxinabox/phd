module Word2Vec
using Base.Collections      # for priority queue
using Base.Cartesian        # for @nexprs
using Distances
using Compat

export LinearClassifier, train_one, GenWordEmbedding, train, accuracy
export save, restore
export find_nearest_words

include("WordDistributions.jl")
include("Trees.jl")
include("WordEmbeddings.jl")
include("WordStreams.jl")
include("Query.jl")
include("Training.jl")

end # module
