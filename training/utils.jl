using Flux.Zygote

function generate_windows(ρ; window_bins)
    ρ_windows = Zygote.Buffer(zeros(Float32, window_bins, length(ρ))) # We use a Zygote Buffer here to keep autodifferentiability
    pad = window_bins ÷ 2 - 1
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
      
function generate_inout(ρ_profiles, c1_profiles, ϕ_profiles; window_width, dx)
    window_bins = 2 * round(Int, window_width / dx) + 1
    ρ_windows_all = Vector{Vector{Float32}}()
    c1_values_all = Vector{Float32}()
    ϕ_functions_all = Vector{Vector{Float32}}()
    for (ρ, c1, ϕ) in zip(ρ_profiles, c1_profiles, ϕ_profiles)
        ρ_windows = generate_windows(ρ; window_bins)
        ϕ_func = generate_phi(ϕ,ρ)
        s = 5
        for i in collect(1:s:length(c1)) 
            if !isfinite(c1[i])
                continue
            end
            push!(ρ_windows_all, ρ_windows[:,i])
            push!(c1_values_all, c1[i])
            push!(ϕ_functions_all, ϕ_func[:,i])
        end
    end
    reduce(hcat, ρ_windows_all), c1_values_all', reduce(hcat, ϕ_functions_all)
end






######### for regularization data



function get_c2_bulk(ρb,ϕ,model)
    window_bins = 401
    model = model |> gpu
    ρ = ones(window_bins)*ρb
    c1s = get_c1_single(model, ρ, ϕ)
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


function ϕ_func(x,params) 
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

function get_g(Φ,T,μ,model)
    L = 30
    V(x) = Φ(x-L/2) + Φ(-(x-L/2))
    xs, ρ =  minimize(L, μ, T, Φ, V, model)
    gr = ρ/ρ[1]
    return xs[1:1000], gr[1501:2500], ρ[1]
end


function Gµneural(µ0,u,model)
V0(x)=0
T = 1
L = 4.1
F(x) = ϕ_func(x,u)
dµ = 0.1
µ = µ0 + dµ
xn,gn1,ρbp = get_g(F,T,μ,model)   
µ = µ0 - dµ
xn,gn2,ρbm = get_g(F,T,μ,model)
dgn = -(gn1*(ρbp[1]^2)-gn2*(ρbm[1]^2))/(2*dµ)
return dgn[1:150]
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

function get_c1_single(model, ρ, ϕ) 
    model = model |> gpu
    ϕ_func = generate_phi(ϕ,ρ)
    ϕ = ϕ_func[:,1] |> gpu 
    function (ρ) 
        input = vcat(ρ, ϕ) |> gpu 
        model(input) |> cpu |> vec  
    end
end


