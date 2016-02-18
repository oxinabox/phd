#!/bin/bash
set -e
shopt -s expand_aliases
cd SOURCE;
alias julia=../UTILITIES/julia/bin/julia
export JULIA_PKGDIR=../UTILITIES/julia_packages

echo "It is strongly recommended that you run this script in steps. You may edit this file to comment out completed steps. The distribution includes all files to be able to run this step-wise.";
echo "Understand that this will likely take days to weeks to complete";
read -p "Proceed? (Enter) - (^C to abort)" TMP;


echo "Running Preprocessing. Input is 1_INPUT, output is 2_PREPROCESSED_OUTPUT";
mv ../2_PREPROCESSED_INPUT ../OLD_2_PREPROCESSED_INPUT;
mkdir ../2_PREPROCESSED_INPUT
julia preprocess_brown_corpus.jl;
julia preprocess_books_corpus.jl;

echo "Running Main Processing. On all cores. Input is 2_PREPROCESSED_OUTPUT, output is 3_OUTPUT";
mv ../3_OUTPUT ../OLD_3_OUTPUT;
mkdir ../3_OUTPUT;
julia -p auto run_sowe2bow.jl brown_glove50;
julia -p auto run_sowe2bow.jl brown_glove100;
julia -p auto run_sowe2bow.jl brown_glove200;
julia -p auto run_sowe2bow.jl brown_glove300;
julia -p auto run_sowe2bow.jl books_glove300;

echo "Running Results Analysis. Input is 3_OUTPUT, output is 4_RESULTS"
mv ../4_RESULTS ../OLD_4_RESULTS;
mkdir ../4_RESULTS
julia results_analysis.jl;
