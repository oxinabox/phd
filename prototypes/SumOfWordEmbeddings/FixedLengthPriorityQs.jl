module FixedLengthPriorityQs

using IntervalHeaps
using Base.Order

export FixedLengthPriorityQueue, length, isempty, dequeue!, peek, enqueue!

type FixedLengthPriorityQueue{ElementType,PriorityType,OrderType<:Ordering}
    priorities::IntervalHeap{PriorityType} #Priorities
    max_len::Int
    ordering::OrderType
    elements:: Dict{PriorityType,Vector{ElementType}}
    
    function FixedLengthPriorityQueue(max_len::Int=2^31, o::OrderType=Forward)
        new(IntervalHeap{PriorityType}(), max_len, o, Dict{PriorityType,Vector{ElementType}}())
    end
    
end

function FixedLengthPriorityQueue{ElementType,PriorityType,OrderType<:Ordering}(
                                    ::Type{ElementType},
                                    ::Type{PriorityType},
                                    max_len::Int=2^31,
                                    o::OrderType=Forward)
    
    FixedLengthPriorityQueue{ElementType,PriorityType,typeof(o)}(max_len,o)    
end

Base.length(pq::FixedLengthPriorityQueue) = length(pq.priorities)
Base.isempty(pq::FixedLengthPriorityQueue) = isempty(pq.priorities)


function dequeue!(pq::FixedLengthPriorityQueue)
    priority = pq.ordering==Forward ? popmin!(pq.priorities) : popmax!(pq.priorities)
    pop!(pq.elements[priority])
end
function peek(pq::FixedLengthPriorityQueue)
    priority = pq.ordering==Forward ? minimum(pq.priorities) : maximum!(pq.priorities)
    pq.elements[priority][end]
end


function enqueue!{ElementType,PriorityType,OrderType}(
                              pq::FixedLengthPriorityQueue{ElementType,PriorityType,OrderType},
                              element::ElementType,
                              priority::PriorityType)
    
    push!(pq.priorities, priority)
    elements_of_priority=get!(pq.elements, priority,  ElementType[])
    push!(elements_of_priority, element)
    if length(pq.priorities)>pq.max_len
        lowest_priority = pq.ordering==Forward ? popmax!(pq.priorities) : popmin!(pq.priorities)
        pop!(pq.elements[lowest_priority])
    end
    @assert(length(pq.priorities)<=pq.max_len)
end

end # module
