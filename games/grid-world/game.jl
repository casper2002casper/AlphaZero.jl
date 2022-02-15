# A simple grid world MDP
# All cells with reward are also terminal
# Environment created by: Zachary Sunberg

using CommonRLInterface
using StaticArrays
using Crayons
using GraphNeuralNetworks

const RL = CommonRLInterface
const SIZE = SA[5,5]
const NUM_N = SIZE[1]*SIZE[2]
p2i(i,j) = ((j-1)*SIZE[1]+i)

function generate_rewards()
  rewards = zeros(Float32, 1, NUM_N)
  rewards[p2i(2,3)] =  1.0
  rewards[p2i(3,3)] =  0.3
  rewards[p2i(2,1)] = -1.0
  rewards[p2i(2,2)] = -0.5
  return rewards
end

function generate_connections()
  connections = Vector{Vector}() #adjacency list of grid
  for j in 1:SIZE[2] 
    for i in 1:SIZE[1]
      point = Vector{Int}()
      for a in [[1,0], [-1,0], [0,1], [0,-1]]
        all([1,1].<=[i,j] + a.<=SIZE) && append!(point,p2i(i+a[1],j+a[2]))
      end
      append!(connections, [point])
    end
  end
  return connections
end  

const REWARDS = generate_rewards()
const CONNECTIONS = generate_connections()

# To avoid episodes of unbounded length, we put an arbitrary limit to the length of an
# episode. Because time is not captured in the state, this introduces a slight bias in
# the value function.
const EPISODE_LENGTH_BOUND = 200

mutable struct World <: AbstractEnv
  position::UInt8
  time :: UInt8
end

function World()
  return World(
    rand(1:NUM_N),
    0)
end

function RL.reset!(env::World)
  env.position = rand(1:NUM_N)
  env.time = 0
end

RL.actions(env::World) = collect(1:NUM_N)
RL.observe(env::World) = env.position
@provide RL.state(env::World) = env.position

RL.terminated(env::World) = !iszero(REWARDS[env.position]) || env.time > EPISODE_LENGTH_BOUND

function RL.act!(env::World, a)
  env.position = a
  env.time += 1
  return REWARDS[env.position]
end

@provide RL.player(env::World) = 1 # An MDP is a one player game
@provide RL.players(env::World) = [1]
#@provide RL.observations(env::World) = [SA[x, y] for x in 1:SIZE[1], y in 1:SIZE[2]]
@provide RL.clone(env::World) = World(env.position, env.time)
@provide RL.setstate!(env::World, state) = (env.position = state)
@provide function RL.valid_action_mask(env::World)
  valid_actions = zeros(Bool, NUM_N)
  valid_nodes = CONNECTIONS[env.position]
  valid_actions[valid_nodes] .= 1
  return valid_actions
end

# Additional functions needed by AlphaZero.jl that are not present in 
# CommonRlInterface.jl. Here, we provide them by overriding some functions from
# GameInterface. An alternative would be to pass them as keyword arguments to
# CommonRLInterfaceWrapper.

function GI.render(env::World)
  for y in reverse(1:SIZE[2])
    for x in 1:SIZE[1]
      s = p2i(x,y)
      r = REWARDS[s]
      if env.position == p2i(x,y)
        c = ("+",)
      elseif r > 0
        c = (crayon"green", "o")
      elseif r < 0
        c = (crayon"red", "o")
      else
        c = (crayon"dark_gray", ".")
      end
      print(c..., " ", crayon"reset")
    end
    println("")
  end
end

function GI.vectorize_state(env::World, state)
  pos = zeros(Float32, 1 , NUM_N)
  pos[state] = 1.0
  return GNNGraph(CONNECTIONS, num_nodes = NUM_N, ndata = [pos;REWARDS])
end

const action_names = ["r", "l", "u", "d"]

function GI.action_string(env::World, a)
  return string(a)
end

# function GI.parse_action(env::World, s)
#   idx = findfirst(==(s), action_names)
#   return isnothing(idx) ? nothing : RL.actions(env)[idx]
# end

function GI.read_state(env::World)
  try
    s = split(readline())
    @assert length(s) == 2
    x = parse(Int, s[1])
    y = parse(Int, s[2])
    @assert 1 <= x <= SIZE[1]
    @assert 1 <= y <= SIZE[2]
    return p2i(x,y)
  catch e
    return nothing
  end
end

GI.heuristic_value(::World) = 0.

GameSpec() = CommonRLInterfaceWrapper.Spec(World())