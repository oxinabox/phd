module Packing
export pack,pack!, unpack, unpack!

function pack{T}(sources::AbstractArray{T}...)
    total_length = sum(map(length,sources))
    package = Vector{T}(total_length)
    pack!(Vector{T}(total_length), sources...)
end

function pack!{T}(output::Vector{T}, sources::AbstractArray{T}...)
    ii = 1
    for s in sources
        output[ii:ii+length(s)-1]=vec(s)
        ii+=length(s)
    end
    output
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