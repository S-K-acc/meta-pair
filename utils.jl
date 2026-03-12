
function generate_windows(ρ; window_bins)
    ρ_windows = Zygote.Buffer(zeros(Float32, window_bins, length(ρ))) # We use a Zygote Buffer here to keep autodifferentiability
    pad = window_bins ÷ 2 - 1 # a number
    ρpad = vcat(ρ[end-pad:end], ρ, ρ[1:1+pad]) 
    for i in 1:length(ρ)
        ρ_windows[:,i] = ρpad[i:i+window_bins-1] 
    end
    copy(ρ_windows)  # copy needed due to Zygote.Buffer
end

function generate_phi(ϕ,ρ)
    ϕ_func = Zygote.Buffer(zeros(Float32, length(ϕ), length(ρ)))
    for i in 1:length(ρ)
        ϕ_func[:,i] = ϕ
    end
    copy(ϕ_func)
end

function get_c2_bulk(ρb,βϕ,model)
    window_bins = 401
    model = model |> gpu
    ρ = ones(window_bins)*ρb
    c1s = get_c1_single(model, ρ, βϕ)
    c2diff = Flux.jacobian(c1s,ρ)[1] / 0.01
    return transpose(c2diff)
end

function get_dc1dϕ_bulk(ρb,βϕ,model;dx = 0.01)
    c1rp = get_c1_singleρϕ(model)
    window_bins = 401 
    rho = Float32.(ρb*ones(401))
    model = model |> gpu
    ρ_window = CuArray(reshape(rho, window_bins, 1))
    c2diff = Flux.jacobian(c1rp,ρ_window,βϕ)[2] / dx
    return c2diff |> cpu |> vec
end

function get_c1_singleρϕ(model)
    model_gpu = gpu(model)

    function (ρ, βϕ)
        ρd  = gpu(ρ)
        βϕd = gpu(βϕ)

        ϕ = reshape(βϕd, 150, 1)         
        input = vcat(ρd, ϕ)               
        model_gpu(input) |> cpu |> vec 
    end
end

function get_c1_neural(model,ϕ) #uses only length of pair potential, not the values
    window_bins = length(model.layers[1].weight[1,:])-size(ϕ)[1] # Get the number of input bins from the shape of the first layer
    model = model |> gpu
    function (ρ, ϕ)
        ρ_windows = generate_windows(ρ; window_bins) |> gpu  
        ϕ_func = generate_phi(ϕ,ρ) |> gpu 
        input = vcat(ρ_windows, ϕ_func)
        model(input) |> cpu |> vec  # Evaluate the model, make sure the result gets back to the CPU, and transpose it to a vector
    end
end



function minimize(L::Number, μ::Number, T::Number, ϕ::Union{<:AbstractArray,<:Function}, Vext::Function, model; ρ0 = nothing, α::Number=0.03, maxiter::Int=10000, dx::Number=0.01, floattype::Type=Float32, tol::Number=max(eps(floattype(1e3)), 1e-8))
    L, μ, T = floattype.((L, μ, T))
    if typeof(ϕ) <: Function 
        βΦ(x) = ϕ(x) / T
        βϕ = floattype.(get_params(βΦ)) # get input number for c1 
    else
        βϕ = floattype.(ϕ./T)
        if maximum(ϕ)>=9 && T != 1
            @warn "∞ finite cutoff scaled with β changed" 
        end
    end
    xs = collect(floattype, dx/2:dx:L)  # Construct the numerical grid
    βVext = Vext.(xs)./T  # Evaluate the external potential on the grid
    infiniteVext = isinf.(βVext)  # Check where Vext is infinite to set ρ = 0 there
    βμ = μ/T

    constant = false #bulk fluid
    if all(==(first(βVext)), βVext)  # Speed up calculations for a bulk fluid
        if ρ0 == nothing
            ρ0 = 0.5
        end
        ρ, ρEL = ρ0, 0
        c1 = get_c1_singleρϕ(model)
        constant = true
    else 
        ρ, ρEL = zero(xs), zero(xs)  # Preallocate the density profile and an intermediate buffer for iteration
        fill!(ρ, 0.5)
        c1 = get_c1_neural(model,βϕ)  # Obtain the c1 functional for the given numerical grid
    end
    i = 0
    while true
        if constant == false
            ρEL .= exp.((βμ .- βVext)  .+ c1(ρ,βϕ))  # Evaluate the RHS of the Euler-Lagrange equation
            ρ .= (1 - α) .* ρ .+ α .* ρEL  # Do a Picard iteration step to update ρ
            ρ[infiniteVext] .= 0  # Set ρ to 0 where Vext = ∞
            clamp!(ρ, 0, Inf)  # Make sure that ρ does not become negative
            Δρmax = maximum(abs.(ρ - ρEL)[.!infiniteVext])  # Calculate the remaining discrepancy to check convergence
        else
            ρEL = exp((βμ - βVext[1])  + c1(ρ*ones(401),βϕ)[1])
            ρ = (1 - α) * ρ + α * ρEL
            Δρmax = abs(ρ - ρEL)
        end
        i += 1
        if Δρmax < tol
            break  # The remaining discrepancy is below the tolerance: break out of the loop and return the result
        end
        if !isfinite(Δρmax) || i >= maxiter
            println("Did not converge (step: $(i) of $(maxiter), ‖Δρ‖: $(Δρmax), tolerance: $(tol))")
            return nothing  # The iteration did not converge, there is no valid result
        end
    end
    if constant 
        return xs, ρ * one.(xs)
    else
        return xs, ρ
    end
end

function µb(ρ::Number,T::Number,ϕ)
    (log(ρ) .- get_c1_neural(model, ϕ/T)(Float32.(ρ*ones(401)),Float32.(ϕ/T))[1])*T
end

function get_χb(ρ,T,ϕ,model)
    cbϕ = get_dc1dϕ_bulk(ρ,Float32.(ϕ./T),model)
    #c2b = get_c2_bulk(ρ,ϕ,model) #alternative method
    #VV = (1/ρ + sum(-c2b*dx))
    prefactor = (1/ρ -c2btilde(ρ,ϕ,T))      
    return cbϕ/(prefactor)
end

function c2btilde(ρ,ϕ,T) # calculate  ̃c₂(k=0) from χµ
    dρ = 0.001
    rho = ρ[1]
    ∂µ∂ρ = (µb(rho+dρ,T,ϕ)-µb(rho-dρ,T,ϕ))/(2*dρ)
    χµ = 1/∂µ∂ρ
    return (χµ-ρ[1]/T)/(χµ*ρ[1]) 
end

function grid2ϕ(x,params) 
    l = length(params)
    Δ = 1.5/(l-1)
    if 0 <= x < 1.5
        bin = x ÷ Δ
        bin = Integer(bin)
        ϵ1 = params[bin+1]
        ϵ2 = params[bin+2]
        return (ϵ2-ϵ1)*(x-bin*Δ)/(Δ) + ϵ1
    else
        return 0
    end
end

function get_params(f::Function)
    grid = collect(0:0.01:1.5-0.01)
    params = f.(grid)
    for i=1:length(params)
        if params[i] > 9 || isinf(params[i]) == true 
            params[i] = 9
        end
    end 
    for i=1:length(params)
        if isnan(params[i]) == true 
            params[i] = 9
        end
    end 
    return params
end

function get_βFexc_funcintegral(xs,model; num_a=100) 
    dx = xs[2] - xs[1]
    da = 1 / num_a
    as = da/2:da:1
    function (ρ,βϕ) 
        if all(==(first(ρ)), ρ)
            c1_function = get_c1_singleρϕ(model)
            aintegral = 0
            for a in as
                aintegral += c1_function(a * ρ[1] .*ones(401), βϕ)[1]
            end
            -ρ[1] * aintegral[1] * dx * da * length(ρ)
        else
            c1_function = get_c1_neural(model, ϕ)
            aintegral = zero(ρ)
            for a in as
                aintegral .+= c1_function(a .* ρ, βϕ)
            end
            -sum(ρ .* aintegral) * dx * da 
        end
    end
end

using Plots; gr()
function plot_χs(χs,vars,cbar_title)
ENV["GR_SUPPRESS_WARNINGS"] = "true"
l = @layout [a{0.95w} b]
cmap = cgrad(:cool)
cmap = cgrad(:cool)             
vmin, vmax = minimum(vars), maximum(vars)
norm(x) = (x - vmin) / (vmax - vmin)
rphi = range(0, 1.5, length=150)
colors = []
for i in 1:size(χs)[2]
    μnorm = clamp(norm(vars[i]), 0, 1)   
    color = cmap[μnorm] 
    push!(colors, color)

end
p1 = plot(rphi,-χs[:,1], c =colors[1],legend=false, xlabel = "r/σ", ylabel = "-χᵩᵇσ²")
for i=2:size(χs)[2]
   plot!(rphi,-χs[:,i], c =colors[i])
end
p2 = heatmap(rand(5,2), clims=(vmin,vmax), framestyle=:none, c=cmap, cbar=true, lims=(-1,0),colorbar_title=cbar_title)
plot(p1, p2, layout=l)
end

function check_normalization(g,L,ρb,ϕ,T; dx = 0.01)
    try
    dρb = 0.01 
    χµ = 1/((µb(ρb +dρb ,T,ϕ)- µb(ρb - dρb ,T,ϕ))/(2*dρb))
    intχµ = χµ*L
    N = ρb*L
    println("<N(N-1)/2>: ",0.5*(-N+N^2+T*intχµ)) 
    # for generic inhomogenous systems e.g.: 0.5*(N²-N+∫dxχµ(x)/β)
    index = Integer(L/(2*dx))
    println("∫drG(r): ",dx*sum(g[1:index])*(ρb^2*L))
    catch e
        "g may not be available for large values of r"
    end
end
