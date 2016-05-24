module ReservoirShuffle

export ReservoirShuffler

"""Shuffling an infinite list in finite memory.
Not perfectly random -- items are likely to remain close to each other, particularly at the start.
The larger the reservoir_size, the better the shuffle is
"""
immutable ReservoirShuffler{B}
    source::B
    reservoir_size::Integer
end


function Base.start(it::ReservoirShuffler)
    reservoir, backer = Base.head_and_tail(it.source, it.reservoir_size)
    (reservoir, backer, Base.start(backer))
end

function Base.done(it::ReservoirShuffler,state)
    reservoir, backer, backer_state = state
    Base.done(backer, backer_state) && length(reservoir)==0
end

function Base.next(it::ReservoirShuffler,state)
    reservoir, backer, backer_state = state
    
    ret_index = rand(1:length(reservoir))
    ret = reservoir[ret_index]
    if Base.done(backer, backer_state) 
        splice!(reservoir, ret_index)
        #Nothing to replace it with
    else
        replacement, backer_state = Base.next(backer, backer_state)
        reservoir[ret_index] = replacement
    end
    ret, (reservoir, backer, backer_state)
end

Base.iteratorsize{B}(::Type{ReservoirShuffler{B}}) = Base.iteratorsize(B)
Base.eltype{B}(::Type{ReservoirShuffler{B}}) = Base.eltype(B)
Base.length(it::ReservoirShuffler) = length(it.source)
Base.size(it::ReservoirShuffler) = size(it.source)

    



end #module