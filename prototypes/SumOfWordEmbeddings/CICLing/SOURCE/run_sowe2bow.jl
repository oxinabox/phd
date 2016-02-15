import JLD
@everywhere using JLD

using Pipe

using Blocks
using Lumberjack


push!(LOAD_PATH, ".")
import sowe2bow
@everywhere using sowe2bow

###################################################################
# Function Definitions

"""
Convience helper function, for use during evaluation.
Takes a data dictionary of embeddings and the target sentence,
which it converts to a bag of words, via the greedy_search method
"""
function greedy_search{S}(data::Dict, target_sent::Vector{S}; kwargs...)
    target = lookup_sowe(data,target_sent)
    greedy_search(data, target; kwargs...)
end

"""
wrappr for jldopen, so that it alway appends, never overwrites
Always append, nev
"""
function jldopen_append(func::Function, filename::AbstractString)
    mode = isfile(filename) ? "r+" : "w" #Only open with "w" if it does't already exist
    jldopen(func, filename, mode)
end


"""
Evaluation Runner.
Farms out the running over all worker processes.
"""
function run(test_set_blocks, save_path="selection.jld")
    try
        Lumberjack.info("Began selection")
        ii = 0 
        map(test_set_blocks) do block
            
            block_res = pmap(block,err_stop=true) do target_sent
                sol, score = greedy_search(data, target_sent, rounds=10_000, log=false)
                (target_sent, sol, score)
            end
            ii+=1
            avg_score = sum([r[3] for r in block_res])/length(block)
            Lumberjack.info("$ii done: $avg_score")
            jldopen_append(save_path) do fh
                write(fh,string(ii), block_res)
            end
            Lumberjack.debug("$ii written to disc")
        end
    catch err
        Lumberjack.error("Unhandled Error", base_exception=err)
    end
    Lumberjack.info("complete selection")
end




#Load Data Helper
#Flexible loading of the data, variety of formats have been used during developtment
@everywhere function flex_load(path)
	raw = load(path)
	if length(keys(raw))>1
		raw
	else
		key = first(keys(raw))
		raw[key]
	end
end


######################################################################
## Running Code
macro constant(varname, varvalue)
  tmp = eval(varvalue)
    quote
        for i in procs()
	      @spawnat i global const $varname = $tmp
        end
    end
end


@everywhere data=0
test_set=0

@constant(corpus_filename,ARGS[1])

if corpus_filename=="books_glove300"
    #Books data stored in slightly different format -- Symbols instead of strings, and seperate file for the corpus

    @everywhere data = load("../2_PREPROCESSED_INPUT/"*corpus_filename*".jld","data_sym")
    test_set = open(deserialize, "../2_PREPROCESSED_INPUT/books_corpus.jsz")
    test_set_blocks = Block(test_set, 1, 1000)
else
    @everywhere data = flex_load("../2_PREPROCESSED_INPUT/"*corpus_filename*".jld")
    #A variety of different keys have been used for the corpus test data during developtment
    corpuskey = filter(k->contains(k,"corpus"), keys(data)) |> first    
    test_set = data[corpuskey]
    test_set_blocks = Block(test_set, 1, 100)
end


#Run Evaluation
add_truck(LumberjackTruck("selection.log"), "file-logger")
run(test_set_blocks, "../3_OUTPUT/$(corpus_filename)_res.jld") 
