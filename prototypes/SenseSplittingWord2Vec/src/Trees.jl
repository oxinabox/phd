"""
Compatible with AbstractTrees.jl
"""
module Trees
export TreeNode, BranchNode, nullnode, leaves_of, internal_nodes_of, average_height, leaves_at_depth, levels
abstract TreeNode

type BranchNode <: TreeNode
    children :: Vector{BranchNode}
    data
end

type NullNode <: TreeNode
end

const nullnode = NullNode()


function Base.show(io::IO, node :: BranchNode)
	print(io, typeof(node)," with ",length(node.children), " children. ", "data = ")
	show(io, node.data)
	nothing
end


Base.getindex(node::BranchNode, idx) = Base.getindex(node.children, idx)
Base.start(node::BranchNode) = Base.start(node.children)
Base.next(node::BranchNode, state) = Base.next(node.children,state)
Base.done(node::BranchNode, state) = Base.done(node.children,state)


isleaf(node::TreeNode) = length(node.children)==0



function leaves_of(root::TreeNode)
    code = Int64[]
    function traverse(node::TreeNode)
        if node == nullnode
            return
        end
        if isleaf(node)
            produce((node.data, copy(code)))    # notice that we should copy the current state of code
        end
        for (index, child) in enumerate(node.children)
            push!(code, index)
            traverse(child)
            pop!(code)
        end
    end
    @task traverse(root)
end

function internal_nodes_of(root::TreeNode)
    function traverse(node::TreeNode)
        if node == nullnode
            return
        end
        if !isleaf(node.children)
            produce(node)
        end
        for child in node.children
            traverse(child)
        end
    end
    @task traverse(root)
end

function average_height(tree::TreeNode)
    (h, c) = (0, 0)
    for (_, path) in leaves_of(tree)
        h += length(path)
        c += 1
    end
    h / c
end



function leaves_at_depth(root::TreeNode, depth)
    function traverse(node::TreeNode, cur_depth::Int)
        @assert(cur_depth<=depth)
        if cur_depth==depth
            if isleaf(node)
                produce(node.data)
            end
            #Else it is a nonleaf and it has some leaf children that we don't care about
        else
            for child in node.children
                traverse(child,cur_depth+1) #Will make produce calls
            end
        end
    end
    @task traverse(root,0)
end


"Returns a 2, vector of vectors. the first is the nodes on each level of the tree, and the second is their codes"
function levels(tree)
    nodes_at_depth = Vector{Vector{BranchNode}}()
    codes_at_depth = Vector{Vector{Vector{Int64}}}()
    
	push!(nodes_at_depth, tree.children) #TODO: fix so works for partiailly empty root node
    push!(codes_at_depth, [[ii] for ii in 1:length(tree.children)])
    
    
    parent_depth = 1
    while true
        #@show parent_depth
        level_nodes = BranchNode[]
        level_codes = Vector{Int64}[]
        
        for (parent_code, parent_node) in zip(codes_at_depth[parent_depth], nodes_at_depth[parent_depth])
            for (ii, child_node) in enumerate(parent_node.children)
                child_code = [parent_code; ii] #Copy and append
                push!(level_nodes, child_node)
                push!(level_codes, child_code)
            end
        end
        
        if length(level_nodes) == 0
            #Done if no nodes on this level, don't need to save empties
            break
        else
            push!(nodes_at_depth, level_nodes)
            push!(codes_at_depth, level_codes)
            parent_depth += 1
        end
    end
    
    nodes_at_depth, codes_at_depth
end

end #module
