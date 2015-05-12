module StandardRAE
using RecursiveAutoencoders


#------------------------------------Tree EVAL

immutable ActData
    c_ij::Embedding
    pp::Embedding
    ĉ_ij::Embedding
end

function zero_col(W::Matrix)
    zeros(size(W,1),1)
end

function tuple_of_cols(a::Matrix)
    @pipe [a[:,col_ii] for col_ii in 1:size(a,2)] |> tuple(_...)
end



function eval_to_tree(rr::RAE, sentence::Words)
    tree = tuple(sentence...)
    cs = eval_word_embeddings(rr, sentence)
    act_tree = tuple_of_cols(cs)
    score_total = 0.0
    while(size(cs,2)>1)
        c_is = cs[:, 1:end-1]
        c_js = cs[:, 2:end]
        
        pps = eval_merges(rr, c_is, c_js)
        ĉ_ijs = unfold_merges(rr,pps)
        scores = eval_scores(rr, c_is, c_js, pps,ĉ_ijs)
        im = indmax(scores)
        
        score_total+=scores[im]
        c_ij=[c_is; c_js][:,im]
        pp = pps[:,im]
        ĉ_ij = ĉ_ijs[:,im]
        act = ActData(c_ij, pp, ĉ_ij)
        act_node = (act_tree[im], act, act_tree[im+1])
        
        cs = [cs[:,1:im-1] pp cs[:,im+2:end]]
        tree = tuple(tree[1:im-1]..., (tree[im], tree[im+1]), tree[im+2:end]...)
        act_tree = tuple(act_tree[1:im-1]..., act_node, act_tree[im+2:end]...)
    end
    
    #Note The final step in tree creates a tuple containing one element, as first and last parts are empty
    tree[1], act_tree[1], cs[:], score_total
end

#------------------------------------BPTS



function BPTS(rae::RAE, nontree::Embedding, δ_above::Matrix)
    #Note a tree. but a terminal state
    (0,0,0,0)
end

function BPTS(rae::RAE, tree::(Any,ActData, Any), δ_above=zero_col(rae.W_e))
    act=tree[2]
    ∇s, δ_input = eval_scores_gradient(rae,act,δ_above)
    δ_left  = δ_input[1:end/2,:]
    δ_right = δ_input[end/2+1:end,:]

    ∇s_left = BPTS(rae, tree[1], δ_left)
    ∇s_right = BPTS(rae, tree[3], δ_right)
    tuple([l+c+r for (c,l,r) in zip(∇s_left,∇s, ∇s_right)]...)
end



function eval_scores_gradient(rae::RAE, 
                              act::ActData,
                              δ_parent=zero_col(rae.W_e))
    #Notice: While this is good to go for multiple concurrent, 
    #It does't actually do so, as a tree is the 
    
    
    c_ijs::Embeddings = act.c_ij''
    pps::Embeddings = act.pp''
    ĉ_ijs::Embeddings = act.ĉ_ij''
    
    #http://neuralnetworksanddeeplearning.com/chap2.html
    N = size(c_ijs,2)
    
    da = (ĉ_ijs - c_ijs)
    dz_d = (1-ĉ_ijs.^2)
    δ_d = da.*dz_d #Output Error

    ∇W_d = 1/N*δ_d*pps'
    ∇b_d = 1/N*sum(δ_d,2)[:]
    
    
    dz_e = (1-pps.^2)
    δ_e = (rae.W_d'*δ_d).*(dz_e .+ δ_parent) #Hidden layer error
    

    ∇W_e = 1/N*δ_e*c_ijs'
    ∇b_e = 1/N*sum(δ_e,2)[:]
    
    ∇s = (∇W_e, ∇b_e, ∇W_d, ∇b_d)
    
    #input error, ie parent error for layer below
    dz_p = (1-c_ijs.^2)
    δ_input = (rae.W_e'*δ_e - da).*dz_p
    
    ∇s, δ_input
end

#--------------UNFOLDING

#tree data in tree is not use, other than it's structure.
#((("the","house"),("destroyed",("the","boy")))  is equivalent to ((("",""),("",("",""))) 
function unfold(rae::RAE, tree::(String,String), pp::Embedding)
    ĉ_is, ĉ_js = reconstruct(rae, pp)
    [ĉ_is ĉ_js]
end


function unfold(rae::RAE, tree::(Any,String), pp::Embedding)
    p̂_is, ĉ_js = reconstruct(rae, pp)
    ĉ_is = unfold(rae, tree[1], p̂_is)
    [ĉ_is ĉ_js]
end

function unfold(rae::RAE, tree::(String,Any), pp::Embedding)
    ĉ_is, p̂_js = reconstruct(rae, pp)
    ĉ_js = unfold(rae, tree[2], p̂_js)
    [ĉ_is ĉ_js]
    
end

function unfold(rae::RAE, tree::(Any,Any), pp::Embedding)
    p̂_is, p̂_js = reconstruct(rae, pp)
    ĉ_is = unfold(rae, tree[1], p̂_is)
    ĉ_js = unfold(rae, tree[2], p̂_js)
    [ĉ_is ĉ_js]
end


end