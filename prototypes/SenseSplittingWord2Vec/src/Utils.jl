module Utils
export save, restore, _pgenerate_gh

# serialize to a file
function save(item, filename::AbstractString)
    open(filename, "w") do fp
        save(item, fp)
    end
end
save(item, fp::IO) = serialize(fp, item)

# restore from a file
function restore(filename::AbstractString)
    open(filename, "r") do fp
        restore(fp)
    end
end
restore(fp::IO) = deserialize(fp)




"https://github.com/JuliaLang/julia/issues/16345"
function _pgenerate_gh(f,c, mname=Symbol("_func_genmagic_hack"*string(rand(1:1024)))::Symbol)
    #Reusing the `mname` in subsequent called can A.) Reclaim memory, B.) Violate certain concurrency expectations
    worker_ids = workers()
    for id in worker_ids
        remotecall_wait(id, mname,f) do mname_i, f_i
            eval(Expr(:global, Expr(Symbol("="), mname_i, f_i)))
        end
    end
        
    worker_pool = WorkerPool(worker_ids)
    
    #Give send a function telling them to look up the function locally
    Base.pgenerate(worker_pool, x->eval(mname)(x), c)  
end

end #module

