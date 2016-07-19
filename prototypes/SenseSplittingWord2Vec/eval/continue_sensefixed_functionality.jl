using WordEmbeddings
using Training
using Query
using Lumberjack
using Utils

test_file = "data/corpora/WikiCorp/tokenised_lowercase_WestburyLab.wikicorp.201004.txt"
load_model_file = "models/ss/keep/tokenised_lowercase_WestburyLab.wikicorp.201004_50__m170000000.model"
progress_count = 170000000
model_file = "models/ss/tokenised_lowercase_WestburyLab.wikicorp.201004_50"
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
		progress_count,
		end_of_iter_callback=save_callback(model_file, "i"),
        end_of_minibatch_callback=save_callback(model_file,"m"),
	)

end

test_word_embedding()
