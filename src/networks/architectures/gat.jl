using GraphNeuralNetworks, Flux

"""
    GatHP

Hyperparameters for the gin architecture.

"""
@kwdef struct GatHP
  depth_common::Int = 3
  depth_phead::Int = 7
  depth_vhead::Int = 5
  hidden_size::Int = 48
end

"""
    Gat <: TwoHeadGraphNeuralNetwork

A simple two-headed GNN architecture with only dense layers.
"""
mutable struct Gat <: GATGraphNeuralNetwork
  gspec
  hyper
  common
  vhead
  phead
end

Flux.@functor Gat (common, vhead, phead)

leakyrelu2(x) = leakyrelu(x, 0.2)

function Gat(gspec::AbstractGameSpec, hyper::GatHP)
  Dense_layers(in, hidden, out, depth, act) = [Dense((i == 1) ? in : hidden, (i == depth) ? out : hidden, act; init = Flux.kaiming_normal(gain=1)) for i in 1:depth]
  GAT_layers(n_size, e_size, depth, act) = [GNNChain(
    GATv2Conv((n_size, e_size) => n_size, act, heads=3, concat=true, add_self_loops=false),
    #LayerNorm(3*n_size),
    Dense_layers(3*n_size, n_size, n_size, 2, act)...) for _ in 1:depth]
  n_size, e_size = GI.state_dim(gspec)
  common = GNNChain(
    #LayerNorm(n_size),
    Dense_layers(n_size, hyper.hidden_size, hyper.hidden_size, 3, selu)...,
    GAT_layers(hyper.hidden_size, e_size, hyper.depth_common, selu)...)
    # x->reshape(x, 1, hyper.hidden_size, :),
    # GroupNorm(hyper.hidden_size, hyper.hidden_size÷8),
    # x->reshape(x, hyper.hidden_size, :))
  vhead = Chain(
    SkipConnection(Chain(Dense_layers(hyper.hidden_size*3 + n_size, hyper.hidden_size*2, hyper.hidden_size, hyper.depth_vhead, selu)...), vcat),
    Dense(hyper.hidden_size*4+n_size, hyper.hidden_size),
    Dense(hyper.hidden_size, 1))
  phead = Chain(
    Dense(hyper.hidden_size*6 + n_size => hyper.hidden_size*5, selu; init = Flux.kaiming_normal(gain=1)),
    Dense(hyper.hidden_size*5 => hyper.hidden_size*4, selu; init = Flux.kaiming_normal(gain=1)),
    Dense(hyper.hidden_size*4 => hyper.hidden_size*3, selu; init = Flux.kaiming_normal(gain=1)),
    Dense_layers(hyper.hidden_size*3, hyper.hidden_size*2, hyper.hidden_size, max(hyper.depth_phead-3, 1), selu)...,
    Dense(hyper.hidden_size, 1, identity))
  return Gat(gspec, hyper, common, vhead, phead)
end

Network.HyperParams(::Type{Gat}) = GatHP

function Base.copy(nn::Gat)
  return Gat(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end