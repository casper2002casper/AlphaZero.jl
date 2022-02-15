import AlphaZero.GI
using StaticArrays
using GraphNeuralNetworks
using Random 

const M = 3 #num machines
const N = 4 #num jobs
const P_MIN = 1#min time
const P_MAX = 5#max time

const S = M*N+2 #[nodes, T, S]
const T = M*N+1

i2mn(i) = CartesianIndices((M,N))[i]

struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv 
  #Problem instance
  process_time::SVector{S, UInt8} #Index in Mxn
  conj_src::SVector{M*N+N, UInt8}
  conj_tar::SVector{M*N+N, UInt8}
  #State
  disj_src::MVector{M*N, UInt8}
  disj_tar::MVector{M*N, UInt8}
  is_done::MVector{S, Bool}
  done_time::MVector{S, UInt16}
  #info
  prev_operation::MVector{N, UInt8}
  prev_machine::MVector{M, UInt8}
end


function generate_conjuctive_edges()
  order = reduce(vcat,[(randperm(M).+i*M) for i in 0:N-1])#flattend order of operations
  target = zeros(UInt8, M*N+N)
  n = 1
  for i in 1:N*M
    if(i%M == 1) #first in row: s->
      target[M*N+n] = order[i]
    end
    if(i%M == 0)#last in row: ->t
      target[order[i]] = T
      n += 1
    else #propagate
      target[order[i]] = order[i+1]
    end
  end
  return target
end

GI.init(::GameSpec) = GameEnv(
  [rand(P_MIN:P_MAX,M*N);0;0],#Nodes time plus t, s = 0
  [collect(1:N*M)..., S * ones(N)...],#From
  generate_conjuctive_edges(),#To
  collect(1:M*N),
  collect(1:M*N),
  [falses(N*M)..., true, true], #[nodes, T, S]
  zeros(UInt16, S),
  collect(-1:N-2) .+ S, #Start node
  S * ones(UInt8, M)
)

GI.init(::GameSpec, s) = GameEnv(
  #Static values
  s.process_time,
  s.conj_src,
  s.conj_tar,
  #Mutable values
  copy(s.disj_src),
  copy(s.disj_tar),
  copy(s.is_done),
  copy(s.done_time),
  copy(s.prev_operation),
  copy(s.prev_machine)
)

GI.spec(::GameEnv) = GameSpec()

GI.two_players(::GameSpec) = false

function GI.set_state!(g::GameEnv, s)
 g.process_time = s.process_time
 g.conj_src = s.conj_src
 g.conj_tar = s.conj_tar
 g.disj_src = s.disj_src
 g.disj_tar = s.disj_tar
 g.is_done = s.is_done
 g.done_time = s.done_time
 g.prev_operation = s.prev_operation
 g.prev_machine = s.prev_machine
end

GI.current_state(g::GameEnv) = (
  process_time = g.process_time,
  conj_src = g.conj_src,
  conj_tar = g.conj_tar,
  disj_src = copy(g.disj_src),
  disj_tar = copy(g.disj_tar),
  is_done = copy(g.is_done),
  done_time = copy(g.done_time),
  prev_operation = copy(g.prev_operation),
  prev_machine = copy(g.prev_machine)
)

GI.white_playing(g::GameEnv) = true

GI.game_terminated(g::GameEnv) = all(g.is_done)

const ACTIONS = collect(1:S)
GI.actions(::GameSpec) = ACTIONS

function GI.actions_mask(g::GameEnv)
  valid_action = zeros(Bool, S)
  valid_action[g.conj_tar[g.prev_operation]] .= true
  valid_action[T] = false #We cant schedule sink node
  return valid_action
end

function GI.play!(g::GameEnv, o)
  #mark operation scheduled
  g.is_done[o] = true
  #previous operation and machine
  mn = i2mn(o)
  k = g.prev_machine[mn[1]] #previous operation done on machine of todo operation
  l = min(g.prev_operation[mn[2]], S) #previous operation done in job of todo operation
  #add disjunctive link
  k != S && (g.disj_tar[k] = o)
  #update done time
  done_time = max(g.done_time[l], g.done_time[k]) + g.process_time[o]
  #update info vectors
  g.prev_machine[mn[1]] = o
  g.prev_operation[mn[2]] = o
  while(true)#propagate expected done time
    g.done_time[o] = done_time = done_time + g.process_time[o]
    o == T && return
    o = g.conj_tar[o]
  end
end

function GI.white_reward(g::GameEnv)
  all(g.is_done) && return maximum(g.done_time)
  return 0
end

function GI.vectorize_state(::GameSpec, state) 
  return  GNNGraph([state.conj_src; state.disj_src], 
                   [state.conj_tar; state.disj_tar], 
                   num_nodes = S, 
                   ndata = hcat(state.done_time, state.is_done)')
end

function GI.symmetries(::GameSpec, s)

end

#####
##### Interaction API
#####

function GI.action_string(::GameSpec, a)

end

function GI.parse_action(::GameSpec, str)

end


function GI.read_state(::GameSpec)

end


function GI.render(g::GameEnv; with_position_names=true, botmargin=true)

end
