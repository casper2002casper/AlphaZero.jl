using GraphNeuralNetworks
using Statistics


"""
  GraphSAGEHP

Hyperparameters for the gin architecture.

"""
@kwdef struct GraphSAGEHP
  depth_common :: Int = 12
  depth_phead :: Int = 3
  depth_vhead :: Int = 3
  hidden_size :: Int = 40
end

"""
    Gin <: TwoHeadGraphNeuralNetwork

A simple two-headed GNN architecture with only dense layers.
"""
mutable struct GraphSAGE <: TwoHeadGraphNeuralNetwork
  gspec
  hyper
  common
  vhead
  phead
end

function GraphSAGE(gspec::AbstractGameSpec, hyper::GraphSAGEHP)
  indim, n_nodes  = GI.state_dim(gspec)
  SAGE_layers(depth) = [GNNChain(SAGEConv(((i==1) ? indim : hyper.hidden_size) => hyper.hidden_size, relu, aggr=mean),
                                BatchNorm(hyper.hidden_size, relu)) for i in 1:depth]
  Dense_layers(size, depth) = [Chain(Dense(size, size, relu), BatchNorm(size)) for _ in 1:depth]
  common = GNNChain(SAGE_layers(hyper.depth_common)...)
  vhead = GNNChain(GlobalPool(mean),  
                   Dense_layers(hyper.hidden_size, hyper.depth_vhead)...,
                   Dense(hyper.hidden_size, 1, identity))
  phead = GNNChain(Parallel(vcat, GNNChain(GlobalPool(mean), x -> repeat(x, 1, n_nodes)), identity),
                  Dense_layers(hyper.hidden_size*2, hyper.depth_phead)...,
                  Dense(hyper.hidden_size*2, 1),
                  x->reshape(x, n_nodes,:),
                  softmax)
  return GraphSAGE(gspec, hyper, common, vhead, phead)
end

Network.HyperParams(::Type{GraphSAGE}) = GraphSAGEHP

function Base.copy(nn::GraphSAGE)
  return GraphSAGE(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end