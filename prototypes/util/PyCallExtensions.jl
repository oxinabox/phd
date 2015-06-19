module PyCallExtensions

using PyCall
import PyCall.@pysym
@pyimport warnings
warnings.filterwarnings("ignore")

export PyObjectIter
#state = inner_inst, next
immutable PyObjectIter
    inner::PyObject 

    function PyObjectIter(obj::PyObject)
        inner = pycall(obj["__iter__"],PyObject)
        new(inner)
    end
end

function next_or_nothing(pyiter::PyObject)
    try
        pycall(pyiter["next"],PyObject)
    catch PyError
        nothing
    end
end


function Base.start(iter::PyObjectIter)
    inner_inst = deepcopy(iter.inner)
    next = next_or_nothing(inner_inst)
    inner_inst, next
end

function Base.next(pyiter::PyObjectIter, state::(PyObject,Union(Nothing,PyObject)))
    inner_inst, next = state
    new_next = next_or_nothing(inner_inst)
    next, (inner_inst,new_next)
end

function Base.done(pyiter::PyObjectIter, state::(PyObject,Union(Nothing,PyObject)))
    inner_inst, next = state
    next==nothing
end



end