#load weights of keras model to julia

using BSON, NPZ, Flux
path = "new_model_weights.npz"
model = Chain(
  Dense(551 => 512, softplus),          # 282_624 parameters
  Dense(512 => 256, softplus),          # 131_328 parameters
  Dense(256 => 128, softplus),          # 32_896 parameters
  Dense(128 => 64, softplus),           # 8_256 parameters
  Dense(64 => 32, softplus),            # 2_080 parameters
  Dense(32 => 1),                       # 33 parameters
)
weights = npzread(path)
function load_weights!(model, weights)
    i = 1
    for layer in model
        if layer isa Dense
            layer.weight .= permutedims(weights["arr_$(i-1)"])  # Keras: (out, in) → Flux: (in, out)
            layer.bias   .= vec(weights["arr_$(i)"])
            i += 2
        end
    end
end
load_weights!(model, weights)
BSON.@save "new_model.bson" model
