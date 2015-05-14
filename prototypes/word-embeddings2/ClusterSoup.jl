#Because of macro hygiene this can not be in a seperate namespace.
#module ClusterSoup
#export chunk_data, prechunked_mapreduce, set_global

using Pipe

function set_global(name::Symbol, value, pid::Int)
    function do_set_global(dummy)
        ex = :(global $name; $name=$value)
        eval(ex)
    end
    remotecall(pid, do_set_global, nothing) 
end

function set_global(name::Symbol, value, pids::Vector{Int64}=workers())
    map(pid->set_global(name,value, pid), workers())
end

function chunk_data(data_name::Symbol, data::Vector)
    chunks = get_chunks(data, nworkers())
    for (pid,chunk) in zip(workers(),chunks)
        println(pid, ": ", typeof(chunk), ": ", length(chunk))
        set_global(data_name, chunk,pid) 
    end
end

function prechunked_mapreduce(data_name::Symbol, map_fun::Function, red_acc::Function)
    function do_mapred(dummy)
        @pipe data_name |> eval |> map(map_fun,_) |> reduce(red_acc, _)
        #reduce(red_acc,map(map_fun,eval(data_name)))
    end
    
    rem_results = @pipe workers() |> map(pid->remotecall(pid,do_mapred, nothing), _)
    @pipe rem_results |> map(fetch,_) |> reduce(red_acc, _)
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

#end