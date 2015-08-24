import nltk
import numpy as np
import itertools 

from glob import glob 
from sys import argv

file_id_key = argv[1]
fileids = glob("Wikipedia_Untagged/*/wiki_"+file_id_key+"*")


for ii,filename in enumerate(fileids):

	if ii % 100 == 0 :
		print("[" +file_id_key+"]" + str(ii) + "/" + str(len(fileids)))

	
	with open(filename,'r') as input_fh:
		text = input_fh.read(-1)
		all_sents=[sent.replace('\n',' ')
				 for sent in nltk.sent_tokenize(text)]

		output_filename = filename.replace("Untagged","Untagged_Tokenized")
		with open(output_filename, 'w') as fh:	
			fh.write('\n'.join(all_sents))
	      
