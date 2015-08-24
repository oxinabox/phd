import numpy as np
import itertools 

from glob import glob 
fileids = glob("Wikipedia_Untagged_Tokenized/*/*")


print("Loading")
all_sents=list(itertools.chain(*[list(open(fileid,'r')) for fileid in fileids]))

print("Loaded")
read_indexes = np.arange(0,len(all_sents)-1)
np.random.shuffle(read_indexes)

output_file_count = 0
output_filename_base = "Wikipedia_Untagged_Shuffled/wiki_sents"
output_filehandle = False
for output_count, read_index in enumerate(read_indexes,0):
    
    if output_count % 1000 == 0:
        output_file_count+=1
        print("Done "+str(output_count) +"/"+str(len(all_sents)))
        if output_filehandle:
            output_filehandle.close()
        output_filehandle = open("%s_%05d.txt" % ( 
					  	 output_filename_base,
						 output_file_count), 'w')
    else:
        output_filehandle.write('\n')
    
    sent = all_sents[read_index]
    output_filehandle.write(sent)

output_filehandle.close()

        
