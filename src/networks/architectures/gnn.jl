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
  depth_common :: Int = 3
  depth_phead :: Int = 1
  depth_vhead :: Int = 1
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
  nin = 1
  nhidden = 10
  ncommon = 5
  hlayers(depth) = [GCNConv(nhidden => nhidden, relu) for i in 1:depth]
  common = GNNChain(GCNConv(nin => nhidden, relu),
                    hlayers(10)...,
                    GCNConv(nhidden => ncommon, relu))
  vhead = GNNChain(GlobalPool(mean),  
            Dense(ncommon, 1), softmax)
  phead = GNNChain(Dense(ncommon, 1), softmax)
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