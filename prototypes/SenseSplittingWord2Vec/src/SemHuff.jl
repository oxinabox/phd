module SemHuff

using BlossomV

import BlossomV.dense_num_edges
"""
dist_func must resturn a values between 0.0 and 1.0. 
result is a matrix where each column is a pairing of items, expressed as there index (1 indexed)
"""
function most_similar_pairings(dist_func::Function, items, consider_nearest_n::Integer)
    m = Matching(length(items), min(dense_num_edges(length(items)), length(items)*consider_nearest_n))
    #sims = Matrix{Int32}(length(items),length(items)).*Inf #for Debug purposes
    for ii in (1:length(items)-1)
        jjs = ii+1:length(items)
        dists = [dist_func(items[ii], items[jj]) for jj in jjs]
        nearest_jjs = if consider_nearest_n < length(jjs)
            jjs[selectperm(dists, 1:consider_nearest_n)]
        else
            jjs
        end
            
        for (jj,dist) in zip(nearest_jjs,dists)
            @assert dist<=1.0
            scale = typemax(Int32)>>4
            approx_dist = round(Int32, dist*scale)
            #sims[ii,jj]=approx_dist
            #println(join([m, ii-1, jj-1, approx_dist],"\t"))
            add_edge(m, ii-1,jj-1 , approx_dist)
        end
    end
    solve(m)
    get_all_matches(m, length(items)) .+ 1    
end

"""
Assumes that the `tree` is already a Huffman tree.
"""
function semhuff(classification_tree, embeddings, consider_nearest_n)
    embedding_dim = length(first(embeddings))
    sim_tree = transform_tree(classification_tree, 
                            leaf_transform = word->word,
                            internal_transform = dummy -> "")
    
    nodes_at_depth, codes_at_depth = levels(sim_tree)
    maxdepth = length(nodes_at_depth)
    
    
    embeds = [embeddings[node.data] for node in nodes_at_depth[maxdepth]]
    #Dict(code => embeddings[node.data] for (code, node) in zip(codes_at_depth[maxdepth], nodes_at_depth[maxdepth]))
    for depth in maxdepth:-1:2
        info("semantically sorting level: $depth")
        nodes = nodes_at_depth[depth]
        
        pair_indexes = most_similar_pairings(Query.angular_dist, embeds, consider_nearest_n)
        
        #We will now, assign the new nodes to parents in arbitary order
        nodes_above = nodes_at_depth[depth-1]
        embeds_above = typeof(embeds)(length(nodes_above))
        pair_jj = 1

        for (above_ii,node_above) in enumerate(nodes_above)

            if Trees.isleaf(node_above)
                embeds_above[above_ii] = embeddings[node_above.data]
            else
                #It is a branch so put a pair of nodes here
                @assert (length(node_above.children) == 2)
                child_index1  = pair_indexes[1,pair_jj]
                child_index2  = pair_indexes[2,pair_jj]
                pair_jj += 1
                node_above.children[1] = nodes[child_index1]
                node_above.children[2] = nodes[child_index2]
                embeds_above[above_ii] = (embeds[child_index1] + embeds[child_index2])/2.0
                #@show nodes[child_index1], nodes[child_index2]
            end
        end
        @assert(pair_jj - 1  == size(pair_indexes,2), "$(pair_jj) != $(length(pair_indexes)) + 1") #All pairs must be assigned.
            
        embeds = embeds_above
    end
    sim_tree
end
    




end #module
