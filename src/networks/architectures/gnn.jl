using GraphNeuralNetworks
using Statistics
"""
    GnnHP

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
@kwdef struct GnnHP
  depth_common :: Int = 5
  depth_phead :: Int = 1
  depth_vhead :: Int = 1
  hidden_size :: Int = 2
  use_batch_norm :: Bool = false
  batch_norm_momentum :: Float32 = 0.6f0
end

"""
    Gnn <: TwoHeadNetwork

A simple two-headed architecture with only dense layers.
"""
mutable struct Gnn <: TwoHeadNetwork
  gspec
  hyper
  common
  vhead
  phead
end

function Gnn(gspec::AbstractGameSpec, hyper::GnnHP)
  indim = 2
  GCN_layers(depth) = [GCNConv(hyper.hidden_size => hyper.hidden_size, relu, add_self_loops=true) for i in 1:depth]
  Dense_layers(depth) = [Dense(hyper.hidden_size, hyper.hidden_size, relu) for i in 1:depth]
  common = GNNChain(GCNConv(indim => hyper.hidden_size, relu, add_self_loops=true),
                    GCN_layers(hyper.depth_common)...,
                    BatchNorm(hyper.hidden_size, relu, momentum=hyper.batch_norm_momentum))
  vhead = GNNChain(Dense_layers(hyper.depth_vhead)...,
                    GlobalPool(mean),  
                    Dense(hyper.hidden_size, 1, tanh))
  phead = GNNChain(Dense_layers(hyper.depth_phead)...,
                  Dense(hyper.hidden_size, 1))
  return Gnn(gspec, hyper, common, vhead, phead)
end

Network.HyperParams(::Type{Gnn}) = GnnHP

function Base.copy(nn::Gnn)
  return Gnn(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end