module SemHuffAdaGram
using SemHuff
using AdaGram
using Trees


function semhuff_initialize_AdaGram(semtree::Trees.BranchNode,
									 word_freqs,
									 dim::Integer,
									 num_meanings::Integer,
									 alpha::Float64=0.01,
									 d::Float64=0.0)
    paths = Dict(node.data => path for (node, path) in Trees.get_paths(semtree))

    codes = Dict(w=> convert(Vector{Int8}, oc - 1) for (w, oc) in leaves_of(semtree))
    dict = AdaGram.Dictionary(convert(Vector{AbstractString},collect(keys(paths))))
    freqs = Vector{Int64}(length(codes))
    huffman_outputs = Vector{AdaGram.HierarchicalOutput}(length(codes))
    for word in dict.id2word
        id = dict.word2id[word]
        freqs[id] = word_freqs[word] 
		 
        huffman_outputs[id] = AdaGram.HierarchicalOutput(codes[word], paths[word])
    end;

    vm = AdaGram.VectorModel(freqs, dim, num_meanings, alpha,d, huffman_outputs)
    vm, dict
end


end #module 
