push!(LOAD_PATH,"../src/")
using Word2Vec
using Base.Test
using Lumberjack
using PooledElements

model_file = ARGS[1]
analogy_file = "data/analogy/questions-words.txt";

filename_only(fn) = fn |> basename |> splitext |> first

testname = filename_only(model_file)*"_"*filename_only(analogy_file)

function analogies(path; preprocess::Function=x->x)
    Task() do 
        for line in open(readlines, path, "r")
            if line[1]==':'
                continue
            end
            a,b,c,d = [word|>preprocess|>pstring for word in split(line)]
            produce(a,b,c,d)
        end
    end
end

@enum RetrievalResult incorrect skip correct

"""
`a` is to `b` as `c` is to `d`
"""
function check_analogy(embed::WordEmbedding, a,b,c,d_ref)
	try 
		d_act = find_nearest_words(embed, "$b-$a+$c", nwords=1)[1][1]"pattern given in http://www.aclweb.org/anthology/N13-1090"
		d_act==d_ref ? correct : incorrect
	catch ee
		if typeof(ee)<:KeyError
			skip
		else
			rethrow()
		end	
	end
end



function test_analogies()
    add_truck(LumberjackTruck(testname*".log"), "filelogger")

	ee = restore(model_file)
	anas = collect( analogies(analogy_file, preprocess=lowercase))
	shuffle!(anas) #Make partway output interesting 
				   #by not always testing same type of analogy in group
	res=RetrievalResult[]
	
	function print_res()
		info("Tested $(length(res)) / $(length(anas))")
		skips = mean(map(r->r==skip, res))*100
		incorrects = mean(map(r->r==incorrect, res))*100
		corrects = mean(map(r->r==correct, res))*100

		info("\tskipped  \t$skips \%")
		info("\tincorrect\t$incorrects \%")
		info("\tcorrect  \t$corrects \%")
	end

	for (ii,ana) in enumerate(anas)
		push!(res, check_analogy(ee, ana...))
		if ii%250==0
			print_res()
		end
	end	
	print_res()
end

test_analogies()
