using GraphNeuralNetworks
using Statistics

"""
    GcnHP

Hyperparameters for the gnn architecture.

| Parameter                     | Description                                  |
|:------------------------------|:---------------------------------------------|
| `width :: Int`                | Number of neurons on each dense layer        |
| `depth_common :: Int`         | Number of dense layers in the trunk          |
| `depth_phead = 1`             | Number of hidden layers in the actions head  |
| `depth_vhead = 1`             | Number of hidden layers in the value  head   |
| `use_batch_norm = false`      | Use batch normalization between each layer   |
| `batch_norm_momentum = 0.6f0` | Momentum of batch norm statistics updates    |
"""
@kwdef struct GcnHP
  depth_common :: Int = 6
  depth_phead :: Int = 3
  depth_vhead :: Int = 3
  hidden_size :: Int = 64
end

"""
    Gin <: TwoHeadNetwork

A simple two-headed architecture with only dense layers.
"""
mutable struct Gcn <: TwoHeadNetwork
  gspec
  hyper
  common
  vhead
  phead
end

function Gcn(gspec::AbstractGameSpec, hyper::GcnHP)
  indim = GI.state_dim(gspec)[1]
  GCN_layers(depth) = [GNNChain(GCNConv((i==1) ? indim : hyper.hidden_size => hyper.hidden_size, relu, add_self_loops=true),
                                BatchNorm(hyper.hidden_size)) for i in 1:depth]
  Dense_layers(depth) = [Dense(hyper.hidden_size, hyper.hidden_size, relu) for i in 1:depth]
  common = GNNChain(GCN_layers(hyper.depth_common)...)
  vhead = GNNChain(GlobalPool(mean),  
                   Dense(hyper.hidden_size, 1, relu))
  phead = GNNChain(Dense_layers(hyper.depth_phead)...,
                  Dense(hyper.hidden_size, 1))
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