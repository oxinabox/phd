using WordEmbeddings
using Training
using Query
using Lumberjack
using Utils

model_dir = "models/ss"
test_filename = "WestburyLab.wikicorp.201004"
test_file = "data/corpora/WikiCorp/tokenised_WestburyLab.wikicorp.201004.txt"


const ndims = 50 
const vname =""
iter_name = "_i0"
base_name  ="$(test_filename)_$(ndims)_$(vname)"
model_file = joinpath(model_dir, base_name)
load_model_file = model_file*iter_name*".model"
log_file = joinpath(model_dir, base_name*".log")
vname*="r"


function test_word_embedding()
    println("=======================================")
    println("Resuming sense splitting word embedding with $(base_name)")
    println("=======================================")
    
    add_truck(LumberjackTruck(log_file, "filelogger"))
	info("restoring $(load_model_file)" )
	embed = restore(load_model_file) 
	info("resuming training")
    @time resume_training!(embed, test_file, 
				end_of_iter_callback=save_callback(model_file)
	)

end

test_word_embedding()
