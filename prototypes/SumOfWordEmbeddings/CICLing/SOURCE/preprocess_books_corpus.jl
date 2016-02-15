##FILE PATHS
corpus_files =["../1_INPUT/books_large_p1.txt", "../1_INPUT/books_large_p2.txt"]
embedding_file = "../1_INPUT/word_embeddings/glove.6B.300d.txt"
output_embedding_data_file = "../2_PREPROCESSED_INPUT/books300d.jld"
output_corpus_file = "../2_PREPROCESSED_INPUT/books_corpus.jsz"

######################

using Pipe
using HDF5

push!(LOAD_PATH, ".")
using WordEmbeddings


############## Books Corpus
println("Loading Books Corpus")

corpus = Vector{Symbol}[]


for fn in corpus_files
    open(fn) do corpus_fh
        for sent in eachline(corpus_fh)
            push!(corpus,[symbol(lowercase(word)) for word in split(sent)])
        end
    end
end

println("Corpus Loaded")
println("\nFinding Vocab")

corpus_vocab = Set{Symbol}() #Symbols are Interned Strings.
for sent in corpus 
    union!(corpus_vocab, sent)
end
corpus_vocab = Set{ASCIIString}(map(string,corpus_vocab))
@show (corpus_vocab |> length)
println("Vocab Found")

println("\nLoading Word Embeddings)
LL, word_indexes, indexed_words =
load_embeddings(embedding_file, length(corpus_vocab), corpus_vocab);
indexed_words = map(Symbol, indexed_words)
word_indexes = Dict([Symbol(word)=>index for (word,index) in word_indexes])
println("Loading Word Embeddings Loaded)

println("\nPruning Corpus, based on Word Embedding Vocab")
known_vocab = Set(indexed_words)
known_corpus = filter(corpus) do sent
    for word in sent
        if !(string(word) in known_vocab)
            return false
        end
    end
    true
end;
@show length(known_corpus)
println("Corpus Pruned")


println("\nSaving Embedding Data (except corpus)")

jldopen(output_embedding_data_file, "w") do file   
    @write file LL
    @write file word_indexes
    @write file indexed_words
    
end

println("Embedding Data Saved")

println("\nSaving test corpus")
subtest_indexes = (rand(length(known_corpus)).<0.01*0.1)  #0.1% chance of being selected to be kept
open(output_corpus_file, "w") do fh
    serialize(fh,known_corpus[subtest_indexes])
end

println("Test Corpus Saved")
println("Done with Preprocessing.")
