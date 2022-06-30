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
  depth_common :: Int = 10
  depth_phead :: Int = 3
  depth_vhead :: Int = 5
  hidden_size :: Int = 32
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
  Dense_layers(in, hidden, out, depth) = [Dense((i==1) ? in : hidden, (i==depth) ? out : hidden, relu) for i in 1:depth]
  f(size) = Chain(Dense_layers(size, size, size, 5)..., 
                  BatchNorm(size, relu))
  GIN_layers(size, depth) = [GNNChain(GINConv(f(size), 0)) for _ in 1:depth]

  indim, _ = GI.state_dim(gspec)
  common = GNNChain(Dense_layers(indim, hyper.hidden_size*2, hyper.hidden_size, 5)...,
                    GIN_layers(hyper.hidden_size, hyper.depth_common)...)
  vhead = GNNChain(GlobalPool(mean),  
                   Dense_layers(hyper.hidden_size, hyper.hidden_size, hyper.hidden_size, hyper.depth_vhead)...,
                   Dense(hyper.hidden_size, 1, σ))
  phead = GNNChain(GlobalConcatenation(mean), 
                   Dense(hyper.hidden_size*2, hyper.hidden_size, relu),
                   GIN_layers(hyper.hidden_size, hyper.depth_phead)...,
                   Dense(hyper.hidden_size, 1))
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