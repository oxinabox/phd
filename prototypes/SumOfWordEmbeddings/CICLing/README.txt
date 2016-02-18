This is the readme for SOWE2BOW.
The full data and code are included hear for reproducible science.
With that said, running the whole process is nontrivial, and this, cut-down, distribution of files is not well tested.
It being a trimmed down, and hopefully more self-contained version of the software used to perform the evaluations.
This code strives for readability over run-ability.

to be completely forth-coming, the only part of this script, in its current form, that has run fully is results_analysis.jl; there could very well be typos in the last few lines of all other files.
To run the whole process in entirity to check it works takes well over a week (potentially several weeks), depending on number of cores. During developtment, parts of it were run at different times, and the results stored in .JLD files -- the very JLD files that are distributed in the data folders.


For those interested in understanding how the method actually works,
 we recommend skipping ever attempting to run this process in its entirity,
and looking only at the file `SOURCE/sowe2bow.jl`, which contains the core algorithm.
This file is truly stand-alone, can could, without adaption, be incorperated as a library into another product, to make use of our algorithm.
The remained of the files are simply there to allow for that algorithm to be tested.

For those wanting to access the data:
Any stored in JLD format is a Julia-type-annotated HDF5 file.
It is best opened using the Julia JLD library, but can reasonably be openned with any HDF5 library, in any programming language.



Requirements
============

Software Requirements
---------------------
This code has only been evaluated on Linux (Ubuntu 14.04). Theoretically all parts work on windows. In practice it is expected to be extremely challanging to run this in windows. The included contents of the UTILITIES folder will not run on windows.

In theory the Julia programming language and all required packages are included.
If however errors occur due to packages not loading.
Then running `sudo ./get_packages.sh` from the UTILITIES folder will attempt to redownload, and reassociate all packages.
After running that command a Julia interactive prompt will open.
enter into this `using JLD`, that will precompile the JLD package. If that goes off without any issues, then everything is theoretically installed correctly.

Beyond these requirements, the Bash shell is used in the various included `*.sh` scripts; and is not included.

 
Note however included is Julia 0.4.3, but testing and development was done on julia 0.5

Hardware Requirements
---------------------
The hardware requirements are quiet high to run this code.
Not so much in the actual running of the algorithm, but in the data processing.

The programming used to preprocess the data, is not clever.
It loads all the sentences into memory.
Run the Preprocessing step, on the Books Corpus, is not recommended on a computer with less than 45Gb of RAM. 
The Books corpus is very large, even for the subset we are using in evaluation;
and has a very large vocabulary, result9ing in relatively slow processing.
Running the runSOWE2BOW.jl on it will take multiple days-weeks, depending on number of cores available.
It is suggested that at least 12 cores be dedicated to the task.

It is for this reason that included in the folders structure are the part way results,
allowing for each to be validated alone.


Folder Structure:
================
The folder structure generally follows the CICLing guidelines.

 -RUN.sh runs the entire process. This includes moving all Data folders (except 1_INPUT) into backups prefixed with "OLD_"
 -unRUN.sh reverses the moving of folder in to backups. It is included to facilitate resuming after aborted uses of RUN.sh

Data folders:
-------------
This is the only place the CICLing guildelines for folder structure are deviated from:
Rather that just including an INPUT and OUTPUT folder, 4 stages of folders are included.

 - 1_INPUT: The raw input to the preprocessing step. Note that the Books Corpus, and the GloVE embeddings are not the contribution, or the property of the authors of this package
 - 2_PREPROCESSED_INPUT: This is the result of running the CorpusLoader code in the 1_INPUT, which carries out the preproccessing described in the Experimental Setup. It packages them in the conviently loadable serialised Julia (.jsz) and HDF5 based JLD formats (.jld)
 - 3_OUTPUT this is the actual output of the algorithm, again in HDF5/JLD format. 
 - 4_RESULTS these are the results produced by running ResAna to anylise the raw output. They are the basis of the tables and charts shown in the paper.


SOURCE
------

The source files contain executable julia source code, in .jl format.

 - `sowe2bow.jl`, which contains the core algorithm.
 - `preprocess_books_corpus.jl` and `preproces_brown_corpus.jl`, for the preprocessing/loading of the Books and Brown corpora respectively.
 - `res_ana.jl` performs the analysis presented in the paper.
 - `run_sowe2bow.jl` runs the core algorithm on data, in parallel etc. It is boilerplate Julia distributed computing code.
 -  `WordEmbedding.jl` a julia module for loading various word-embedding file formats.

For those wanting to access the data:
Any stored in JLD format is a Julia-type-annotated HDF5 file.
It is best opened using the Julia JLD library,
but can reasonably be openned with any HDF5 library, in any programming language.



