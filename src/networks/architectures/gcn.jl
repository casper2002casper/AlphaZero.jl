using GraphNeuralNetworks
using Statistics

"""
    GcnHP

Hyperparameters for the gcn architecture.


"""

@kwdef struct GcnHP
  depth_common :: Int = 6
  depth_phead :: Int = 2
  depth_vhead :: Int = 2
  hidden_size :: Int = 64
end

"""
    Gin <: TwoHeadGraphNeuralNetwork

A simple two-headed GNN architecture with only dense layers.
"""
mutable struct Gcn <: TwoHeadGraphNeuralNetwork
  gspec
  hyper
  common
  vhead
  phead
end

function Gcn(gspec::AbstractGameSpec, hyper::GcnHP)
  indim, n_nodes  = GI.state_dim(gspec)
  GCN_layers(depth) = [GNNChain(GCNConv(((i==1) ? indim : hyper.hidden_size) => hyper.hidden_size, relu, add_self_loops=true),
                                BatchNorm(hyper.hidden_size, relu)) for i in 1:depth]
  Dense_layers(size, depth) = [Chain(Dense(size, size, relu), BatchNorm(size)) for _ in 1:depth]
  common = GNNChain(GCN_layers(hyper.depth_common)..., BatchNorm(hyper.hidden_size))
  vhead = GNNChain(GlobalPool(mean),  
                   Dense_layers(hyper.hidden_size, hyper.depth_vhead)...,
                   Dense(hyper.hidden_size, 1, identity))
  phead = GNNChain(Parallel(vcat, GNNChain(GlobalPool(mean), x -> repeat(x, 1, n_nodes)), identity),
                  Dense_layers(hyper.hidden_size*2, hyper.depth_phead)...,
                  Dense(hyper.hidden_size*2, 1),
                  x->reshape(x, n_nodes,:),
                  softmax)
  return Gcn(gspec, hyper, common, vhead, phead)
end

Network.HyperParams(::Type{Gcn}) = GcnHP

function Base.copy(nn::Gcn)
  return Gcn(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end