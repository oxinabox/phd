using Pipe
using DataStructures
using JLD
using DataFrames

push!(LOAD_PATH, ".")
using DataStructuresExtended


OUTPUT_DIR = "../4_RESULTS/"


##################################
# Evaluation Metrics

function jaccard_index(aa::DataStructures.Accumulator,bb::DataStructures.Accumulator)
    sum(intersect(aa,bb))/sum(union(aa,bb))
end

function precision(actual::DataStructures.Accumulator,reference::DataStructures.Accumulator)
    sum(intersect(actual,reference))/sum(actual)
end
function recall(actual::DataStructures.Accumulator,reference::DataStructures.Accumulator)
    sum(intersect(actual,reference))/sum(reference)
end

function f1_score(actual::DataStructures.Accumulator,reference::DataStructures.Accumulator)
    prec = precision(actual,reference)
    rec = recall(actual,reference)
    2*prec*rec/(prec+rec)
end

#################################
# Data Loading
function flatten_raw_blocked_records{T,V}(records::Dict{T,V})  
    key_types = @pipe records |> keys|> map(parse,_) |> map(typeof,_) |> union
    @assert length(key_types)==1
    key_type =  key_types[1]
    
    rec_types = @pipe records |> values |> map(typeof,_) |> union
    rec_type = length(rec_types)==1 ? rec_types[1] : V
    ret = rec_type() #Construct it, it is some form of vector, or we will error soon
    sizehint!(ret) = sum([length(r) for r in values(records)])
    
    sorted_keys = @pipe records |> keys |> collect |> sort
    for key in sorted_keys
        push!(ret, records[key]...)
    end
    ret
end
"""
Loads results files, whether they are in blocks, or not
"""
function multi_load(path)
    raw = load(path)
    if length(raw)==1
        raw[keys(raw)|> first]
    else
        flatten_raw_blocked_records(raw)
    end
end



###################################
# Analysis Output Writing
"""
The columns that have score info
"""
function get_score_column_names(bag_res)
    @pipe (bag_res.colindex |> keys |> map(string,_)
                        |>filter(col -> contains(col,"jaccard") 
                                    ||  contains(col,"perfect") 
                                    ||  contains(col,"recall")
                                    ||  contains(col,"precision")
                                    ||  contains(col,"f1" ),_)
                        |> map(Symbol,_))
end



"""
Score every result from the experiments.
Warning: raw_names is resolved at runtime to global variables, and also used to produce names for the columns
"""
function get_bag_res(raw_names)
    bag_res = DataFrame()
    bag_res[:ground] = [counter(rset[1]) for rset in eval(symbol(first(raw_names)))]
    bag_res[:ground_len] = Int[sum(ss) for ss in bag_res[:ground]]
    exp_names = ASCIIString[]
    for raw_name in raw_names 
        colname = join(split(raw_name,"_")[1:end-1],"_")
        push!(exp_names, colname)
        raw = eval(raw_name|>Symbol)
        bag_res[colname*"_actual"|> Symbol] = [counter(rset[2]) for rset in raw]
        #bag_res[colname*"_distance"|> Symbol] = Float64[-1*rset[3] for rset in raw]
    end

    for exp_name in exp_names
        actual = bag_res[exp_name*"_actual" |> symbol]
        bag_res[exp_name*"_perfect" |> symbol] = actual.==bag_res[:ground]
        bag_res[exp_name*"_jaccard" |> symbol] = map(jaccard_index, actual, bag_res[:ground])
        bag_res[exp_name*"_precision" |> symbol] = map(precision, actual, bag_res[:ground])
        bag_res[exp_name*"_recall" |> symbol] = map(recall, actual, bag_res[:ground])
        bag_res[exp_name*"_f1" |> symbol] = map(f1_score, actual, bag_res[:ground])
    end
    
    bag_res
end



"""
Helper version of mean, that skips NaNs, Infs, -Infs etc
"""
function mean_of_finite_elements(list)
    mean(list[isfinite(list)])
end

"""
Records the overall mean scores
"""
function write_overall(bag_res, savename)
    keep_cols = get_score_column_names(bag_res)
    open(OUTPUT_DIR*savename*".csv","w") do fh
        for keep_col in keep_cols
            col_res = bag_res[keep_col]
            mean_res = mean_of_finite_elements(col_res)
            line = string(keep_col)*"_mean\t"*string(mean_res)*"\n"
            print(line)
            write(fh,line)
        end
    end
end


"""
Records the scores for each different length of sentence.
"""
function write_len_scores(bag_res, savename)
    keep_cols = get_score_column_names(bag_res)
    
    len_scores = aggregate(bag_res[[:ground_len, keep_cols...]], :ground_len, mean_of_finite_elements)
    sort!(len_scores)
    writetable(OUTPUT_DIR*savename*".csv", len_scores)
    len_scores
end


#################################
# Run Code
println("Loading results")

brown_glove300_raw = multi_load("../3_OUTPUT/brown_glove300_res.jld")
println("loaded brown_glove300_raw")
brown_glove200_raw = multi_load("../3_OUTPUT/brown_glove200_res.jld")
println("loaded brown_glove200_raw")
brown_glove100_raw = multi_load("../3_OUTPUT/brown_glove100_res.jld")
println("loaded brown_glove100_raw")
brown_glove50_raw = multi_load("../3_OUTPUT/brown_glove50_res.jld")
println("loaded brown_glove50_raw ")

books_glove300_raw = multi_load("../3_OUTPUT/books_glove300_res.jld")

println("Raw results loaded")
#####
#
println("\nDoing Analysis of Books")
books_bag_res = get_bag_res(["books_glove300_raw"])
write_overall(books_bag_res, "overall_books_corpus_glove300")
write_len_scores(books_bag_res, "at_length_books_corpus_glove300")

println("\n\nDoing Analysis of Brown")
brown_bag_res = get_bag_res(["brown_glove50_raw","brown_glove100_raw", "brown_glove200_raw", "brown_glove300_raw"])
write_overall(brown_bag_res, "overall_brown_corpus")
write_len_scores(brown_bag_res, "at_length_brown_corpus")

println("All Analysis complete")

