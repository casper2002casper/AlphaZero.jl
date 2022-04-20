using GraphNeuralNetworks
using Statistics:mean
using NNlib:gather

struct GlobalConcatenation{F} <: GNNLayer
  aggr::F
end

function (l::GlobalConcatenation)(g::GNNGraph, x::AbstractArray)
  d = reduce_nodes(l.aggr, g, x)
  d = gather(d, graph_indicator(g))
  return vcat(x, d)
end

(l::GlobalConcatenation)(g::GNNGraph) = GNNGraph(g, gdata=l(g, node_features(g)))

"""
    GinHP

Hyperparameters for the gin architecture.

"""
@kwdef struct GinHP
  depth_common :: Int = 12
  depth_phead :: Int = 3
  depth_vhead :: Int = 3
  hidden_size :: Int = 64
end

"""
    Gin <: TwoHeadGraphNeuralNetwork

A simple two-headed GNN architecture with only dense layers.
"""
mutable struct Gin <: TwoHeadGraphNeuralNetwork
  gspec
  hyper
  common
  vhead
  phead
end

function Gin(gspec::AbstractGameSpec, hyper::GinHP)
  Dense_layers(size, depth) = [Dense(size, size, relu) for _ in 1:depth]
  f(in, out) = Chain(Dense(in, out, relu),
                    Dense_layers(out, 2)..., 
                    BatchNorm(out, relu))
  GIN_layers(in, out, depth) = [GNNChain(GINConv(f(i==1 ? in : out, out), 0)) for i in 1:depth]

  indim, _ = GI.state_dim(gspec)
  common = GNNChain(GIN_layers(indim, hyper.hidden_size, 3)...)
  vhead = GNNChain(GlobalPool(mean),  
                   Dense_layers(hyper.hidden_size, 2)...,
                   Dense(hyper.hidden_size, 1, identity))
  phead = GNNChain(GlobalConcatenation(mean), 
                   GIN_layers(2*hyper.hidden_size, 32, 2)...,
                   Dense(32, 1))
  return Gin(gspec, hyper, common, vhead, phead)
end

Network.HyperParams(::Type{Gin}) = GinHP

function Base.copy(nn::Gin)
  return Gin(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end