module Utils
using JLD
export save, restore, _pgenerate_gh, orderless_equivalent, ≅, @param_save, importfrom


function importfrom(moduleinstance::Module, functionname::Symbol, argtypes::Tuple)
    meths = methods(moduleinstance.(functionname), argtypes)
    importfrom(moduleinstance, functionname, meths)
end

"""
eg `importfrom(CorpusLoaders.Semcor, :sensekey)`
"""
function importfrom(moduleinstance::Module, functionname::Symbol)
    meths = methods(moduleinstance.(functionname))
    importfrom(moduleinstance, functionname, meths)
end

"""
Import a method from a function, into the current module (eg Main).
Useful if using two modules both exporting (and thus failing to export)
the same name.
"""
function importfrom(moduleinstance::Module, functionname::Symbol, meths::Base.MethodList)
    for mt in meths
        paramnames = collect(mt.lambda_template.slotnames[2:end])
        paramtypes = collect(mt.sig.parameters[2:end])
        paramsig = ((n,t)->Expr(:(::),n,t)).(paramnames, paramtypes)

        funcdec = Expr(:(=),
			Expr(:call, functionname, paramsig...),
			Expr(:call, :($moduleinstance.$functionname), paramnames...)
        )
        current_module().eval(funcdec) #Runs at global scope
    end
end


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
	if nprocs()==1
		return Base.AsyncGenerator(f,c)
	end
	#warn("_pgenerate_gh is deprecated now that it is theoreticall fixed in master")
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


#################################################


function names_candidates(blk::Expr)
    names_in_block = Vector{Symbol}()
    for a in blk.args
        typeof(a) <: Expr || continue
        if a.head == :(=)
            push!(names_in_block, a.args[1])
        else #Recurse, so we captured things in blocks or behind `const`
            append!(names_in_block, names_candidates(a))
        end
    end
    names_in_block
end

macro param_save(filename, blk::Expr)
    names_in_block =  names_candidates(blk)   

    quote
        $(esc(blk))
        names_defined = Set($(names_in_block)) ∩ Set(names(current_module()))
        names_and_vals =[(string(name), current_module().eval(name)) for name in names_defined] 
										#Got to eval with the eval from current module

        println("Paramaters -- saving to $($(esc(filename)))")
        println("----------")
        println(join(("$n = $v" for (n,v) in names_and_vals),"\n"))
        JLD.save($(esc(filename)), Base.flatten(names_and_vals)...)
        println("----------")
	 end
end

end #module

