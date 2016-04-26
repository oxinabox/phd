module WordDistributions
using Lumberjack
using WordStreams

#TODO Types in this are all screwy. It doesn't really matter, but it would be lcaerer is Floats were not largely being used for Ints

export get_distribution, strip_infrequent, compute_frequency, word_distribution

function get_distribution(corpus_fileio::IO)
    distribution = Dict{AbstractString,Float64}()
    word_count = 0

    for i in words_of(corpus_fileio)
        if !haskey(distribution, i)
            distribution[i] = 1
        else
            distribution[i] += 1
        end
        word_count += 1
    end

    (word_count, distribution)
end

function get_distribution(corpus_filename::AbstractString)
    open(corpus_filename, "r") do fs
        return get_distribution(fs)
    end
end

function strip_infrequent(distribution::Dict{AbstractString,Float64}, min_count::Int)
    stripped_distr = Dict{AbstractString,Float64}()
    word_count = 0

    for (k,v) in distribution
        if v >= min_count
            word_count += Int(round(v))
            stripped_distr[k] = v
        end
    end

    (word_count, stripped_distr)
end


function compute_frequency{S<:AbstractString,C<:Number}(distribution_counts::Dict{S,C}, word_count::Int)

	distribution=Dict{S,Float64}()
	for (k, v) in distribution_counts
        distribution[k] = v/word_count
    end
	distribution
end

function word_distribution(source::Union{AbstractString, IO}, min_count::Int=5)
    tic()

    info("Finding word distribution...")
    word_count, distribution = get_distribution(source)
    info("Word Count: $word_count, Vocabulary Size: $(length(keys(distribution)))")

    info("Stripping infrequent words...")
    word_count, distribution = strip_infrequent(distribution, min_count)
    info("Word Count: $word_count, Vocabulary Size: $(length(keys(distribution)))")

	distribution = compute_frequency(distribution, word_count)
    info("Compute time: ", toq())

    distribution
end


end #end Module
