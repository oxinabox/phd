module Packing
export pack, unpack, unpack!

function pack{T}(sources::AbstractArray{T}...)
    total_length = sum(map(length,sources))
    package = Vector{T}(total_length)
    
    ii = 1
    for s in sources
        package[ii:ii+length(s)-1]=vec(s)
        ii+=length(s)
    end
    package
end

function unpack!{T}(package::Vector{T}, dests::AbstractArray{T}...)
    ii = 1
    for s in dests
        s[:]= package[ii:ii+length(s)-1]
        ii+=length(s)
    end
    dests
end

function unpack{T}(package::Vector{T}, sizes::Union{Int,Tuple{Vararg{Int}}}...)
    ds = [Array{T}(sz) for sz in sizes]
    unpack!(package, ds...)
end

end#module