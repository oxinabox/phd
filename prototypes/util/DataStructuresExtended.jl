module DataStructuresExtended
using DataStructures
export convert, counter, ==,hash



function Base.convert{T}(::Type{Matrix{T}}, q::Deque{Vector{T}})
    n_cols = 0
    block = q.head
    for _ in 1:q.nblocks
        n_cols+=length(block)
        block=block.next
    end
    n_rows = length(front(q))
    data=Matrix{T}(n_rows,n_cols)
    
    block = q.head
    data_offset = 0 
    for _ in 1:q.nblocks
        for block_col in block.front:block.back
            @inbounds data[:,data_offset+block_col] = block.data[block_col]
            
        end
        data_offset += length(block)
        block=block.next
        
    end
    data
end



function DataStructures.counter(T::DataType,seq)
    ct = counter(T)
    for x::T in seq
        push!(ct, x)
    end
    return ct
end

function =={K,V}(lhs::DataStructures.Accumulator{K,V},rhs::DataStructures.Accumulator{K,V})
    lhs.map == rhs.map    
end


function Base.hash{K,V}(obj::DataStructures.Accumulator{K,V},h::UInt64)
    hash(obj.map,h)
end

function Base.union{K,V}(aa::DataStructures.Accumulator{K,V},bb::DataStructures.Accumulator{K,V})
    ret = DataStructures.Accumulator(K,V)
    for key in union(keys(aa), keys(bb))
        ret.map[key]=max(aa[key], bb[key])
    end
    ret
end

function Base.intersect{K,V}(aa::DataStructures.Accumulator{K,V},bb::DataStructures.Accumulator{K,V})
    ret = DataStructures.Accumulator(K,V)
    for key in union(keys(aa), keys(bb))
        ret.map[key]=min(aa[key], bb[key])
    end
    ret
end

function Base.sum{K,V}(aa::DataStructures.Accumulator{K,V})
    sum(aa.map |> values)
end

end #end module