"""
This module provides utilities to build neural networks with Flux,
along with a library of standard architectures.
"""
module FluxLib

export SimpleNet, SimpleNetHP, ResNet, ResNetHP, Gcn, GcnHP, Gin, GinHP, GraphSAGE, GraphSAGEHP

using ..AlphaZero

using CUDA
using Base: @kwdef
#using Statistics: var

import Flux
import GraphNeuralNetworks

CUDA.allowscalar(false)
array_on_gpu(::Array) = false
array_on_gpu(::CuArray) = true
array_on_gpu(arr) = error("Usupported array type: ", typeof(arr))

using Flux: relu, softmax, flatten, σ
using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection, Parallel
using GraphNeuralNetworks: GCNConv, SAGEConv
import Zygote

function Flux.unbatch(g::GraphNeuralNetworks.GNNGraph, x) 
  changes = g.graph_indicator[1:end-1] .!= g.graph_indicator[2:end] 
  index =  [0; Array(findall(changes)); length(g.graph_indicator)]
  return [x[index[i]+1:index[i+1]] for i in 1:g.num_graphs]
end

#####
##### Flux Networks
#####

"""
    FluxNetwork <: AbstractNetwork

Abstract type for neural networks implemented using the _Flux_ framework.

The `regularized_params_` function must be overrided for all layers containing
parameters that are subject to regularization.

Provided that the above holds, `FluxNetwork` implements the full
network interface with the following exceptions:
[`Network.HyperParams`](@ref), [`Network.hyperparams`](@ref),
[`Network.forward`](@ref) and [`Network.on_gpu`](@ref).
"""
abstract type FluxNetwork <: AbstractNetwork end

function Base.copy(nn::Net) where Net <: FluxNetwork
  #new = Net(Network.hyperparams(nn))
  #Flux.loadparams!(new, Flux.params(nn))
  #return new
  return Base.deepcopy(nn)
end

Network.to_cpu(nn::FluxNetwork) = Flux.cpu(nn)

function Network.to_gpu(nn::FluxNetwork)
  CUDA.allowscalar(false)
  return Flux.gpu(nn)
end

function Network.set_test_mode!(nn::FluxNetwork, mode)
  Flux.testmode!(nn, mode)
end

Network.convert_input(nn::FluxNetwork, x) =
  Network.on_gpu(nn) ? Flux.gpu(x) : x

Network.convert_output(nn::FluxNetwork, x) = Flux.cpu(x)

Network.params(nn::FluxNetwork) = [Flux.params(nn)]

Network.set_params!(nn::FluxNetwork, weights) = Flux.loadparams!(nn, weights[1])

# This should be included in Flux
function lossgrads(f, args...)
  val, back = Zygote.pullback(f, args...)
  grad = back(Zygote.sensitivity(val))
  return val, grad
end

function Network.train!(callback, nn::FluxNetwork, opt::Adam, loss, data, n)
  optimiser = Flux.ADAM(opt.lr)
  params = Flux.params(nn)
  for (i, d) in enumerate(data)
    d = Network.convert_input_tuple(nn, d)
    l, grads = lossgrads(params) do
      loss(d...)
    end
    Flux.update!(optimiser, params, grads)
    callback(i, l)
  end
end

function Network.train!(
    callback, nn::FluxNetwork, opt::CyclicNesterov, loss, data, n)
  lr = CyclicSchedule(
    opt.lr_base,
    opt.lr_high,
    opt.lr_low, n=n)
  momentum = CyclicSchedule(
    opt.momentum_high,
    opt.momentum_low,
    opt.momentum_high, n=n)
  optimiser = Flux.Nesterov(opt.lr_low, opt.momentum_high)
  params = Flux.params(nn)
  for (i, d) in enumerate(data)
    l, grads = lossgrads(params) do
      loss(d...)
    end
    Flux.update!(optimiser, params, grads)
    optimiser.eta = lr[i]
    optimiser.rho = momentum[i]
    callback(i, l)
  end
end

regularized_params_(l) = []
regularized_params_(l::GraphNeuralNetworks.GCNConv) = [l.weight, l.bias]
regularized_params_(l::GraphNeuralNetworks.SAGEConv) = [l.weight, l.bias]
regularized_params_(l::Flux.Dense) = [l.weight]
regularized_params_(l::Flux.Conv) = [l.weight]

function Network.regularized_params(net::FluxNetwork)
  return (w for l in Flux.modules(net) for w in regularized_params_(l))
end

function Network.gc(::FluxNetwork)
  GC.gc(true)
  # CUDA.reclaim()
end

#####
##### Common functions between two-head neural networks
#####

"""
    TwoHeadNetwork <: FluxNetwork

An abstract type for two-head neural networks implemented with Flux.

Subtypes are assumed to have fields
`hyper`, `gspec`, `common`, `vhead` and `phead`. Based on those, an implementation
is provided for [`Network.hyperparams`](@ref), [`Network.game_spec`](@ref),
[`Network.forward`](@ref) and [`Network.on_gpu`](@ref), leaving only
[`Network.HyperParams`](@ref) to be implemented.
"""
abstract type TwoHeadNetwork <: FluxNetwork end

abstract type TwoHeadGraphNeuralNetwork <: TwoHeadNetwork end

Network.params(nn::TwoHeadNetwork) = [Flux.params(nn.common), Flux.params(nn.vhead), Flux.params(nn.phead)]

function Network.set_params!(nn::TwoHeadNetwork, weights) 
  Flux.loadparams!(nn.common,weights[1])
  Flux.loadparams!(nn.vhead, weights[2])
  Flux.loadparams!(nn.phead, weights[3])
end

function Network.forward(nn::TwoHeadGraphNeuralNetwork, state)
  c = nn.common(state, state.ndata.x)
  v = nn.vhead(state, c)
  p = nn.phead(state, c)
  p = softmax_nodes(state, p)
  p = Flux.unbatch(state, p)
  return (p, v)
end

function Network.forward(nn::TwoHeadNetwork, state)
  c = nn.common(state)
  v = nn.vhead(c)
  p = nn.phead(c)
  return (p, v)
end

# Flux.@functor does not work with abstract types
function Flux.functor(nn::Net) where Net <: TwoHeadNetwork
  children = (nn.common, nn.vhead, nn.phead)
  constructor = cs -> Net(nn.gspec, nn.hyper, cs...)
  return (children, constructor)
end

Network.hyperparams(nn::TwoHeadNetwork) = nn.hyper

Network.game_spec(nn::TwoHeadNetwork) = nn.gspec

Network.on_gpu(nn::TwoHeadNetwork) = array_on_gpu(nn.vhead[end].bias)

#####
##### Include networks library
#####

include("architectures/simplenet.jl")
include("architectures/resnet.jl")
include("architectures/gcn.jl")
include("architectures/gin.jl")
include("architectures/graphSAGE.jl")
end
