


function initialize_embedding(embed::WordSenseEmbedding, randomly::RandomInited)
    for i in embed.vocabulary
        embed.embedding[i] = rand(embed.dimension,1) * 2 - 1
    end
    embed
end




