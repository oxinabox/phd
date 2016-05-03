module Training

using Base.Collections
using SoftmaxClassifier
using WordStreams
using WordDistributions
using Trees
using WordEmbeddings
using Lumberjack
using Query


export train, random_init, run_training!, initialize_embedding

include("Training/general_training.jl")
include("Training/word_embedding_training.jl")
include("Training/wordsense_embedding_training.jl")

end
