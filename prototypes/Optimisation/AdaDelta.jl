module AdaDelta
export adadelta

################################ From Optim base ##########################
using Optim
import Optim.OptimizationTrace
import Optim.MultivariateOptimizationResults
import Optim.update!

function maxdiff(x::Array, y::Array)
    res = 0.0
    for i in 1:length(x)
        delta = abs(x[i] - y[i])
        if delta > res
            res = delta
        end
    end
    return res
end

function assess_convergence(x::Array,
                            x_previous::Array,
                            f_x::Real,
                            f_x_previous::Real,
                            gr::Array,
                            xtol::Real,
                            ftol::Real,
                            grtol::Real)
    x_converged, f_converged, gr_converged = false, false, false

    if maxdiff(x, x_previous) < xtol
        x_converged = true
    end

    # Absolute Tolerance
    # if abs(f_x - f_x_previous) < ftol
    # Relative Tolerance
    if abs(f_x - f_x_previous) / (abs(f_x) + ftol) < ftol #|| nextfloat(f_x) >= f_x_previous
        f_converged = true
    end

    if norm(vec(gr), Inf) < grtol
        gr_converged = true
    end

    converged = x_converged || f_converged || gr_converged

    return x_converged, f_converged, gr_converged, converged
end

macro gdtrace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(gr)
            end
            grnorm = norm(gr, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    grnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end
end

################################### New Content #####################

"""AdaDelta algorithm from [this paper](http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf)

Note:

 - Does not use a linesearch, 
    - all the Optim Algorithms that do use linesearch make many calls to `d.fg` per iteration
    - this only does one, therefor can afford to do many more iterations (in same amount of time).
 - has two hyper parameters, the decay constant: ρ and the smoothing constant ϵ
    - not highly sensitive to their values.
"""
function adadelta{T}(d::Union{DifferentiableFunction, TwiceDifferentiableFunction},
                             initial_x::Array{T};
                             xtol::Real = 1e-32,
                             ftol::Real = 1e-8,
                             grtol::Real = 1e-8,
                             iterations::Integer = 20_000,
                             store_trace::Bool = false,
                             show_trace::Bool = false,
                             extended_trace::Bool = false,
                             ρ = 0.95, #decay constant
                             ϵ = 1e-6 #smoothing constant    
    )
    
    @assert 0.0<ρ<1.0
    @assert 0.0<ϵ
    
    function rms(x²)
        √(x².+ϵ)
    end
    
    function update_running_squared_average!(avg, g)
        avg[:].*=ρ
        avg[:].+=(1-ρ).*(g.^2)
    end

    # Maintain current state in x
    x = copy(initial_x)
    x_previous = copy(initial_x)

    # Track calls to function and gradient
    f_calls, g_calls = 0, 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in gr
    gr = similar(x)
    

    #Running windows of past gradient and delta
    E_gr² = zeros(size(x))
    E_Δx² = zeros(size(x))


    # Store f(x) in f_x
    f_x_previous = NaN
    f_x = d.fg!(x, gr)
    
    f_calls+=1
    g_calls+=1

    f_x_best = f_x
    x_best = x
    
    # TODO: How should this flag be set?
    mayterminate = false

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace || extended_trace
    

    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = false, false, false

    converged = false
    iteration = 0
    @gdtrace
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1
        
        update_running_squared_average!(E_gr², gr)
        Δx = -rms(E_Δx²)./rms(E_gr²) .* gr 
        update_running_squared_average!(E_Δx², Δx)
        # Update current position
        x_previous = x
        x+=Δx
        
        # Update the function value and gradient
        f_x_previous = f_x
        f_x = d.fg!(x, gr)
        f_calls += 1
        g_calls += 1

        if f_x<=f_x_best
            f_x_best = f_x
            x_best=x
        end
        
        
        x_converged,
        f_converged,
        gr_converged,
        converged = assess_convergence(x,
                                       x_previous,
                                       f_x,
                                       f_x_previous,
                                       gr,
                                       xtol,
                                       ftol,
                                       grtol)

        @gdtrace
    end

    return MultivariateOptimizationResults("AdaDelta",
                                           initial_x,
                                           x_best,
                                           f_x_best,
                                           iteration,
                                           iteration >= iterations,
                                           x_converged,
                                           xtol,
                                           f_converged,
                                           ftol,
                                           gr_converged,
                                           grtol,
                                           tr,
                                           f_calls,
                                           g_calls)
end



end #end module