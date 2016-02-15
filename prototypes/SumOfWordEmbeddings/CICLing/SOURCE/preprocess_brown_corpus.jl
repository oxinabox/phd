##FILE PATHS
embedding_files =  ["../1_INPUT/word_embeddings/glove.6B.50d.txt",
                    "../1_INPUT/word_embeddings/glove.6B.100d.txt",
                    "../1_INPUT/word_embeddings/glove.6B.200d.txt",
                    "../1_INPUT/word_embeddings/glove.6B.300d.txt"]

output__files =    ["../2_PREPROCESSED_INPUT/brown_glove50.jld",
                    "../2_PREPROCESSED_INPUT/brown_glove100.jld",
                    "../2_PREPROCESSED_INPUT/brown_glove200.jld",
                    "../2_PREPROCESSED_INPUT/brown_glove300.jld"]
 
######################
using Pipe
using HDF5

push!(LOAD_PATH, ".")
using WordEmbeddings


############## brown Corpus
println("Loading Brown Corpus")

dir_prefix="../1_INPUT/brown/"
corpus_files = [dir_prefix*filename for filename in filter(x->length(x)==4, readdir(dir_prefix))]
corpus = Vector{Symbol}[]
for fn in corpus_files
    open(fn) do corpus_fh
        for sent in eachline(corpus_fh)
            tokens = split(sent)
            if length(tokens)==0
                continue
            end
            words = [split(token,"/")[1] for token in tokens]
            push!(corpus,[symbol(lowercase(word)) for word in words])
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

println("\nAbout to load the embeddings: ", join(embedding_files,",_") )
println("\n")

for (output_file, embedding_file) in zip(output_files, embedding_files)
    println("\nLoading Word Embeddings: ", embedding_file)
    LL, word_indexes, indexed_words =
	load_embeddings(embedding_file, length(corpus_vocab), corpus_vocab);
    indexed_words = map(Symbol, indexed_words)
    word_indexes = Dict([Symbol(word)=>index for (word,index) in word_indexes])
    println("Loading Word Embeddings Loaded")

    println("\nPruning Corpus, based on Word Embedding Vocab")
    known_vocab = Set(indexed_words) 
    test_corpus = filter(corpus) do sent 
        for word in sent
            if !(string(word) in known_vocab)
                return false
            end
        end
        true
    end;
    @show length(test_corpus)
    println("Corpus Pruned")
    
    
    println("\nSaving Embedding Data")
    jldopen(output_file, "w") do file   
        @write file LL
        @write file word_indexes
        @write file indexed_words
        @write file test_corpus
    end
    println("Embedding Data Saved")
end

println("Done with Preprocessing/Loading Brown")
