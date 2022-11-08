"""
This module provides utilities to build neural networks with Flux,
along with a library of standard architectures.
"""
module FluxLib

export SimpleNet, SimpleNetHP, ResNet, ResNetHP, Gcn, GcnHP, Gin, GinHP, Gat, GatHP, GraphSAGE, GraphSAGEHP

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

using Flux: relu, softmax, flatten, σ, cpu
using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection, Parallel
using GraphNeuralNetworks: GCNConv, SAGEConv
using NNlib: scatter
import Zygote

function unbatch(graph_indicator, x, num_graphs)
  length(x) == 1 && return [x]
  changes = graph_indicator[1:end-1] .!= graph_indicator[2:end]
  index = [0; Array(findall(changes)); length(graph_indicator)]
  return [x[index[i]+1:index[i+1]] for i in 1:num_graphs]
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

function Base.copy(nn::Net) where {Net<:FluxNetwork}
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
  CUDA.memory_status()
  optimiser = Flux.ADAM(opt.lr)
  params = Flux.params(nn)
  for (i, d) in enumerate(data)
    d = Network.convert_input_tuple(nn, d)
    l, grads = lossgrads(params) do
      loss(d...)
    end
    Flux.update!(optimiser, params, grads)
    CUDA.memory_status()
    GC.gc(true)
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
    d = Network.convert_input_tuple(nn, d)
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
regularized_params_(l::GraphNeuralNetworks.GATv2Conv) = [regularized_params_(l.dense_i)..., regularized_params_(l.dense_j)..., regularized_params_(l.dense_e)..., l.bias, l.a]
regularized_params_(l::Flux.Dense) = [l.weight, l.bias]
regularized_params_(l::Flux.Conv) = [l.weight]
regularized_params_(l::Flux.BatchNorm) = [l.β, l.γ]

function Network.regularized_params(net::FluxNetwork)
  return (w for l in Flux.modules(net) for w in regularized_params_(l))
end

function Network.gc(::FluxNetwork)
  GC.gc(true)
  CUDA.reclaim()
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

abstract type GATGraphNeuralNetwork <: TwoHeadGraphNeuralNetwork end

Network.params(nn::TwoHeadNetwork) = [Flux.params(nn.common), Flux.params(nn.vhead), Flux.params(nn.phead)]

function Network.set_params!(nn::TwoHeadNetwork, weights)
  Flux.loadparams!(nn.common, weights[1])
  Flux.loadparams!(nn.vhead, weights[2])
  Flux.loadparams!(nn.phead, weights[3])
end

function Network.forward(nn::GATGraphNeuralNetwork, g)
  c = nn.common(g, g.ndata.x)
  is_machine = g.ndata.x[2, :] .== 1.0
  is_vehicle = g.ndata.x[3, :] .== 1.0
  is_next_op = g.ndata.x[4, :] .== 1.0
  next_op_nodes = findall(is_next_op)

  A = adjacency_matrix(g, Bool, nodes = is_next_op)

  M = A .& is_machine'
  V = A .& is_vehicle'

  index = vcat([fill(o, sum(M[o,:]) * sum(V[o,:])) for o in eachindex(next_op_nodes)]...)
  machine_index = vcat([repeat(findall(M[o,:]), inner=sum(V[o,:])) for o in eachindex(next_op_nodes)]...)
  vehicle_index = vcat([repeat(findall(V[o,:]), outer=sum(M[o,:])) for o in eachindex(next_op_nodes)]...)
  operation_index = next_op_nodes[index]
  graph_index = g.graph_indicator[operation_index]
  
  g_data = vcat(
    scatter(mean, c[:, is_next_op], g.graph_indicator[is_next_op]),
    scatter(mean, c[:, is_machine], g.graph_indicator[is_machine]),
    scatter(mean, c[:, is_vehicle], g.graph_indicator[is_vehicle]))
  p_data = vcat(
    c[:, operation_index],
    c[:, machine_index],
    c[:, vehicle_index],
    g_data[:, graph_index])
  
  v = nn.vhead(g_data)
  p = nn.phead(p_data)
  p = unbatch(graph_index, p, g.num_graphs)
  p = softmax.(p)
  return (p, v)
end

function Network.forward(nn::TwoHeadGraphNeuralNetwork, state)
  c = nn.common(state, state.ndata.x)
  v = nn.vhead(state, c)
  p = nn.phead(state, c)
  is_next_action = state.ndata.x[6, :] .== 1
  p = unbatch(state.graph_indicator[is_next_action], p[is_next_action], state.num_graphs)
  p = softmax.(p)
  return (p, v)
end

function Network.forward(nn::TwoHeadNetwork, state)
  c = nn.common(state)
  v = nn.vhead(c)
  p = nn.phead(c)
  return (p, v)
end

# Flux.@functor does not work with abstract types
function Flux.functor(nn::Net) where {Net<:TwoHeadNetwork}
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
include("architectures/gat.jl")
include("architectures/graphSAGE.jl")
end
