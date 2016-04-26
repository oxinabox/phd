module WordDistributions

using Lumberjack
using WordStreams

export get_distribution, strip_infrequent, compute_frequency!, word_distribution

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

function compute_frequency!(distribution::Dict{AbstractString,Float64}, word_count::Int)
    for (k, v) in distribution
        distribution[k] /= word_count
    end
    nothing
end

function word_distribution(source::AbstractString, min_count::Int=5)
    tic()

    info("Finding word distribution...")
    word_count, distribution = get_distribution(source)
    info("Word Count: $word_count, Vocabulary Size: $(length(keys(distribution)))")

    info("Stripping infrequent words...")
    word_count, distribution = strip_infrequent(distribution, min_count)
    info("Word Count: $word_count, Vocabulary Size: $(length(keys(distribution)))")

    compute_frequency!(distribution, word_count)
    info("Compute time: ", toq())

    distribution
end


end #end Module
