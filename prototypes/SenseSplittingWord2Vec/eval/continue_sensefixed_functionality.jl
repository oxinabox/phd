using WordEmbeddings
using Training
using Query
using Lumberjack
using Utils

test_file = "data/corpora/WikiCorp/tokenised_lowercase_WestburyLab.wikicorp.201004.txt"
load_model_file = "models/ss/tokenised_lowercase_WestburyLab.wikicorp.201004_300__m0.model"
model_file = "models/ss/tokenised_lowercase_WestburyLab.wikicorp.201004_300"
log_file = model_file*".log"


function test_word_embedding()
    println("=======================================")
    println("Resuming sense splitting word embedding")
    println("=======================================")
    
    add_truck(LumberjackTruck(log_file, "filelogger"))
	info("restoring $(load_model_file)" )
	embed = restore(load_model_file) 
	info("resuming training")
    @time resume_training!(embed, test_file, 
        end_of_iter_callback=save_callback(model_file, "i"),
        end_of_minibatch_callback=save_callback(model_file,"m"),
	)

end

test_word_embedding()
