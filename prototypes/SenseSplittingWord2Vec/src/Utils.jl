module Utils
export save, restore, _pgenerate_gh, orderless_equivalent, ≅

# serialize to a file
function save(item, filename::String)
    open(filename, "w") do fp
        save(item, fp)
    end
end
save(item, fp::IO) = serialize(fp, item)

# restore from a file
function restore(filename::String)
    open(filename, "r") do fp
        restore(fp)
    end
end
restore(fp::IO) = deserialize(fp)




"https://github.com/JuliaLang/julia/issues/16345"
function _pgenerate_gh(f,c, mname=Symbol("_func_genmagic_hack"*string(rand(UInt64)))::Symbol)
    #Reusing the `mname` in subsequent called can A.) Reclaim memory, B.) Violate certain concurrency expectations
    worker_ids = workers()

	s_f_buff = IOBuffer()
	serialize(s_f_buff, f)
	s_f = s_f_buff.data

    function make_global(mname_i, s_f_i)
        eval(Expr(:global, 
					Expr(Symbol("="), mname_i, 
						Expr(:call,:deserialize,
							Expr(:call, :IOBuffer, s_f_i)
							)
						)
				 )
			)
    end
   
    @sync for id in worker_ids
        @async remotecall_wait(make_global, id, mname, s_f)
    end
    worker_pool = WorkerPool(worker_ids)

    #Give send a function telling them to look up the function locally
    Base.pgenerate(worker_pool, x->eval(mname)(x), c)  
end

"""Performs unordered nested equivelences check, using provided equivelence operator.
"""
function orderless_equivalent(equiv_op, lhs,rhs)
    if equiv_op(lhs,rhs)
        return true
    else
        if !applicable(length, lhs) || !applicable(length, rhs)
            return false
        end
        if length(lhs)!=length(rhs)
            return false
        end
        if length(lhs)==1
            return equiv_op(first(lhs),first(rhs)) 
        end
            
        matched = BitVector(length(rhs))
        
        for ll in lhs            
            success=false
            @inbounds for (ii,rr) in enumerate(rhs)
                if !matched[ii] && orderless_equivalent(equiv_op,ll,rr)
                    matched[ii]=true
                    success=true
                    break
                end
            end
            if !success
                return false
            end
        end
        return all(matched)
    end
end

"""
Performed unordered equivalence check using default eqality (==)
Eg ([[2,1],3,3]≅[3,3,[1,2]])"""
≅(lhs,rhs) = orderless_equivalent(==, lhs,rhs)


end #module

