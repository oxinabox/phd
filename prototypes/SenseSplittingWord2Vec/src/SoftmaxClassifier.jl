module SoftmaxClassifier
using StatsFuns


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

@fastmath @inline function softmax2{R<:Number}(t1::R,t2::R)
	#This is the softmax function, but particularly optimised.
    #Note you must in the expondent subtract the larger from the smaller, so you get underfloat (to zero) rather than overflow (to Inf) so that things are all behaved. https://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
    
    if t1>t2
        z=t2-t1
        m  = exp(z)
        r1 = inv(one(R)+m)
        r2 = one(R) - r1
        (r1,r2)
    else #t2>t1
        z=t1-t2
        m  = exp(z)
        r2 = inv(one(R)+m)
        r1 = one(R) - r2
    end
    (r1,r2)
end

@fastmath function predict{F<:Number}(c::LinearClassifier{2},
											 x::AbstractVector{F})
    t1=zero(F)
    t2=zero(F)
	# i.e. (t1,t2) = (c.weights'*x)...
    @inbounds for ii in 1: size(c.weights,1)
        @inbounds t1+=c.weights[ii,1]*x[ii]
        @inbounds t2+=c.weights[ii,2]*x[ii]
    end
    return softmax2(t1,t2)
end


##############Normal Path (profiling found that using generated functions did not help)

function predict{F<:Number,K}(c::LinearClassifier{K}, x::AbstractVector{F})
    return softmax(c.weights'*x)
end


function train_one!{F<:Number,K}(
					c::LinearClassifier{K},
					x::AbstractVector{F},
					y::Int64,
					α::Number=0.025f0)
    # if !in(y, 1 : K)
    #     msg = @sprintf "A sample is discarded because the label y = %d is not in range of 1 to %d" y K
    #     warn(msg)
    #     return
    # end

	outputs = collect(predict(c, x))
    outputs[y] -= 1

    # c.weights -= α * x' * outputs;
    # BLAS.ger!(-α, vec(x), c.outputs, c.weights)
    for i in 1:K
        m = α * outputs[i]
        for j in 1:c.n
            c.weights[j, i] -= m * x[j]
        end
    end
end

function train_one!{F1<:Number, F2<:Number,K}(
					c::LinearClassifier{K},
					x::AbstractVector{F1},
					y::Int64,
					input_gradient::AbstractVector{F2},
					α::Number=0.025f0)
	#Using Cross Entropy Error
	#result is
	#input_gradient += -ForwardDiff.gradient(log(collect(predict(c, x))[y]), x)
	
	#@assert(!(x |> isnan |> any))
	#@assert(!(c.weights[:] |> isnan |> any))
	outputs = collect(predict(c, x))
    outputs[y] -= 1
    
	#@assert(!(outputs |> isnan |> any))
	# input_gradient = ( c.weights * outputs' )'
    # BLAS.gemv!('N', α, c.weights, c.outputs, 1.0, input_gradient)
    for i in 1:K
        m = α * outputs[i]
        for j in 1:c.n
            input_gradient[j] += m * c.weights[j, i]
        end
    end

    # c.weights -= α * x' * outputs;
    # BLAS.ger!(-α, vec(x), c.outputs, c.weights
    for i in 1:K
        m = α * outputs[i]
        for j in 1:c.n
			c.weights[j, i] -= m * x[j]
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
function accuracy{F<:Number,K}(c::LinearClassifier{K}, X::AbstractMatrix{F}, y::AbstractVector{Int64})
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
