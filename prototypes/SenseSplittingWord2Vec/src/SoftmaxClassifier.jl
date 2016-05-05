module SoftmaxClassifier
using StatsFuns
using Base.Cartesian        # for @nexprs


export LinearClassifier, predict, train_one!, accuracy, log_likelihood

"""
linear softmax classifier (with stochastic gradient descent)
K is number of outputs
"""
type LinearClassifier{K}
    #K is number of outputs
    n::Int64 # number of inputs
    weights::Array{Float32, 2} # n * k weight matrix

end

function LinearClassifier(k, n)
    weights = rand(n, k) * 2 - 1; # range is [-1, 1]
    LinearClassifier{k}(n, weights)
end


###Carefully Optimised K=2 Path. (a lot of Profiling went into this)

@fastmath @inline function softmax2{R<:AbstractFloat}(t1::R,t2::R)
	#This is the softmax function, but particularly optimised.
    #Note you must in the expondent subtract the larger from the smaller, so you get underfloat (to zero) rather than overflow (to Inf) so that things are all behaved. https://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
    
    if t1>t2
        z=t2-t1
        m  = exp(z)
        r1 = inv(one(R)+m)
        r2 = m*r1
        (r1,r2)
    else #t2>t1
        z=t1-t2
        m  = exp(z)
        r2 = inv(one(R)+m)
        r1 = m*r2
    end
    (r1,r2)
end

@fastmath function predict{F<:AbstractFloat}(c::LinearClassifier{2},
											 x::AbstractVector{F})
    t1=zero(F)
    t2=zero(F)
    @inbounds for ii in 1: size(c.weights,1)
        @inbounds t1+=c.weights[ii,1]*x[ii]
        @inbounds t2+=c.weights[ii,2]*x[ii]
    end
    return softmax2(t1,t2)
end


##############Normal Path (profiling found that using generated functions did not help)

function predict{F<:AbstractFloat,K}(c::LinearClassifier{K}, x::AbstractVector{F})
    return softmax(c.weights'*x)
end


function train_one!{F<:AbstractFloat,K}(
					c::LinearClassifier{K},
					x::AbstractVector{F},
					y::Int64,
					α::AbstractFloat=0.025f0)
    # if !in(y, 1 : K)
    #     msg = @sprintf "A sample is discarded because the label y = %d is not in range of 1 to %d" y K
    #     warn(msg)
    #     return
    # end

	outputs = collect(predict(c, x))
    outputs[y] -= 1

    # c.weights -= α * x' * outputs;
    # BLAS.ger!(-α, vec(x), c.outputs, c.weights)
    m = 0.0
    j = 0
    limit = c.n - 4
    for i in 1:K
        m = α * outputs[i]
        j = 1
        while j <= limit
            @nexprs 4 (idx->c.weights[j + idx - 1, i] -= m * x[j + idx - 1])
            j+=4
        end
        while j <= c.n
            c.weights[j, i] -= m * x[j]
            j+=1
        end
    end
end

function train_one!{F1<:AbstractFloat, F2<:AbstractFloat,K}(
					c::LinearClassifier{K},
					x::AbstractVector{F1},
					y::Int64,
					input_gradient::AbstractVector{F2},
					α::AbstractFloat=0.025f0)
	#@assert(!(x |> isnan |> any))
	#@assert(!(c.weights[:] |> isnan |> any))
	outputs = collect(predict(c, x))
    outputs[y] -= 1
    
	#@assert(!(outputs |> isnan |> any))
	# input_gradient = ( c.weights * outputs' )'
    # BLAS.gemv!('N', α, c.weights, c.outputs, 1.0, input_gradient)
    m = 0.0
    j = 0
    limit = c.n - 4
    for i in 1:K
        m = α * outputs[i]
        j = 1
        while j <= limit
            @nexprs 4 (idx->input_gradient[j+idx-1] += m * c.weights[j+idx-1, i])
            j+=4
        end
        while j <= c.n
            input_gradient[j] += m * c.weights[j, i]
            j+=1
        end
    end

    # c.weights -= α * x' * outputs;
    # BLAS.ger!(-α, vec(x), c.outputs, c.weights)
    for i in 1:K
        m = α * outputs[i]
        j = 1
        while j <= limit
            @nexprs 4 (idx->c.weights[j + idx - 1, i] -= m * x[j + idx - 1])
            j+=4
        end
        while j <= c.n
            c.weights[j, i] -= m * x[j]
            j+=1
        end
    end
	#@assert(!(c.weights[:] |> isnan |> any))
end

# calculate the overall log likelihood. Mainly used for debugging
function log_likelihood(c, X, y)
    n = size(X, 1)
    l = 0
    for i in 1:n
        l += log(predict(c, X[i, :])[y[i]])
    end
    return l
end

# calculate the accuracy on the testing dataset
function accuracy{F<:AbstractFloat,K}(c::LinearClassifier{K}, X::AbstractMatrix{F}, y::AbstractVector{Int64})
    n = size(X, 1)
    succ = 0
    for i in 1 : n
        output = predict(c, X[i, :])
        if maximum(output) == output[y[i]]
            succ += 1
        end
    end
    return succ / n
end



end #module
