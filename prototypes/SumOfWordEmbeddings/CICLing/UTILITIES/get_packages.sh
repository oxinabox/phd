#!/bin/bash
set -e

export JULIA_PKGDIR=./julia_packages
julia/bin/julia -e "Pkg.init()"
julia/bin/julia -e "Pkg.add(\"Pipe\");Pkg.add(\"Lumberjack\");Pkg.add(\"DataStructures\");Pkg.add(\"Blocks\");"
julia/bin/julia -e "Pkg.add(\"DataFrames\")"
julia/bin/julia -e "Pkg.add(\"JLD\")"

cd "julia_packages/v0.4/Blosc/deps/"
export JULIA_PKGDIR=../../../../julia_packages
../../../../julia/bin/julia build.jl
cd ../../HDF5/deps/
../../../../julia/bin/julia build.jl

cd ../../../..
sudo chmod -R 777 .
export JULIA_PKGDIR=./julia_packages
julia/bin/julia
