
module ClusterSoup
#export r_chunk_data, prechunked_mapreduce, Base.put!, update_remote, fetch_reduce

using Pipe

import Base.put!
function Base.put!(pids::Vector{Int}, val) 
    [put!(RemoteRef(id)::RemoteRef, val) for id in pids] 
end



function update_remote(rr::RemoteRef, updater!::Function)
    function update!()
        @pipe rr |> fetch |> updater!(_)
    end
    remotecall(rr.where, update!) 
end

function r_chunk_data(data::Vector)
    all_chuncks = get_chunks(data, nworkers()) |> collect;
    remote_chunks = [put!(RemoteRef(pid)::RemoteRef, all_chuncks[ii]) for (ii,pid) in enumerate(workers())]
    #Have to add the type annotation sas otherwise it thinks that, RemoteRef(pid) might return a RemoteValue
end



function fetch_reduce(red_acc::Function, rem_results::Vector{RemoteRef})
    total = nothing 
    #TODO: consider strongly wrapping total in a lock, when in 0.4, so that it is garenteed safe 
    @sync for rr in rem_results
        function gather(rr)
            res=fetch(rr)
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

function prechunked_mapreduce(r_chunks::Vector{RemoteRef}, map_fun::Function, red_acc::Function)
    rem_results = map(r_chunks) do rchunk
        function do_mapred()
            @assert r_chunk.where==myid()
            @pipe r_chunk |> fetch |> map(map_fun,_) |> reduce(red_acc, _)
        end
        remotecall(r_chunk.where,do_mapred)
    end
    @pipe rem_results|> convert(Vector{RemoteRef},_) |> fetch_reduce(red_acc, _)
end

function prechunked_mapreduce(r_chunks::Vector{RemoteRef}, r_map_funs::Vector{RemoteRef}, red_acc::Function)
    rem_results = map(zip(r_chunks,r_map_funs)) do rs
        r_chunk=rs[1]
        r_map_fun=rs[2]
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