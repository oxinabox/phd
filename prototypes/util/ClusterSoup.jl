
module ClusterSoup
export scatter_data, prescattered_mapreduce, put!, update_remote,replace_remote, fetch_reduce, rmap

using Pipe
using Zlib

import Base.put!
function put!(pids::Vector{Int}, val) 
    RemoteRef[put!(RemoteRef(id)::RemoteRef, val) for id in pids] 
end


#put! now with compression,
#compression from 0 (fastest, no compression), to 9 (slowest, most compression)
#It is a tradeoff between increasing CPU time (higher compression level) and increasing network time (lower compression level)
#Requires Zlib and Type T to be define at all processes
function put!{T}(pids::Vector{Int}, data::T, compression_level=5)
    data_streamed = IOBuffer()
    serialize(data_streamed, data)
    const data_ser_compressed = compress(data_streamed.data, compression_level)
    
    function decomp(comp_data::Array{UInt8,1}) 
       data_ser = decompress(comp_data)
       deserialize(IOBuffer(data_ser)) :: T
    end
    
    RemoteRef[remotecall(pid, decomp, data_ser_compressed) for pid in pids]
end
    

function rmap(fun::Function, r_refs::Vector{RemoteRef})
    RemoteRef[remotecall(r_ref.where, fun, r_ref) for r_ref in r_refs]
end

function update_remote(updater!::Function,rr::RemoteRef)
    function update!()
        @pipe rr |> fetch |> updater!(_)
        rr
    end
    remotecall(rr.where, update!) 
end

function replace_remote(updater!::Function,rr::RemoteRef)
    function update!()
        @pipe rr |> take! |> updater!(_)
        rr
    end
    remotecall(rr.where, update!) 
end


function scatter_data(data::Vector)
    all_chuncks = get_chunks(data, nworkers()) |> collect;
    remote_chunks = RemoteRef[put!(RemoteRef(pid), all_chuncks[ii]) for (ii,pid) in enumerate(workers())]
    #Have to add the type annotation sas otherwise it thinks that, RemoteRef(pid) might return a RemoteValue
end



function fetch_reduce(red_acc::Function, rem_results::Vector{RemoteRef})
    total = nothing 
    #TODO: consider strongly wrapping total in a lock, when in 0.4, so that it is garenteed safe 
    @sync for rr in rem_results
        function gather(rr)
            const res=fetch(rr)
            if total===nothing
                total=res
            else 
                total=red_acc(total,res)
            end
        end
        @async gather(rr)
    end
    total
end

function prescattered_mapreduce(r_chunks::Vector{RemoteRef}, map_fun::Function, red_acc::Function)
    rem_results = map(r_chunks) do rchunk
        function do_mapred()
            @assert r_chunk.where==myid()
            @pipe r_chunk |> fetch |> map(map_fun,_) |> reduce(red_acc, _)
        end
        remotecall(r_chunk.where,do_mapred)
    end
    @pipe rem_results|> convert(Vector{RemoteRef},_) |> fetch_reduce(red_acc, _)
end

function prescattered_mapreduce(r_chunks::Vector{RemoteRef}, r_map_funs::Vector{RemoteRef}, red_acc::Function)
    rem_results = map(zip(r_chunks,r_map_funs)) do rs
        const r_chunk, r_map_fun=rs
        @assert r_map_fun.where==r_chunk.where
        
        function do_mapred()
            @assert r_chunk.where==myid()
            map_fun = fetch(r_map_fun)
            @pipe r_chunk |> fetch |> map(map_fun,_) |> reduce(red_acc, _)
        end
        remotecall(r_chunk.where,do_mapred) 
    end
    @pipe rem_results|> convert(Vector{RemoteRef},_) |> fetch_reduce(red_acc, _)
end


function get_chunks(data::Vector, nchunks::Int)
    base_len, remainder = divrem(length(data),nchunks)
    chunk_len = fill(base_len,nchunks)
    chunk_len[1:remainder]+=1 #remained will always be less than nchunks
    function _it() 
        for ii in 1:nchunks
            chunk_start = sum(chunk_len[1:ii-1])+1
            chunk_end = chunk_start + chunk_len[ii] -1
            chunk = data[chunk_start:chunk_end]
            produce(chunk)
        end
    end
    Task(_it)
end

end