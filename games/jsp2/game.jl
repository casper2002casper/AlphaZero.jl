import AlphaZero.GI
using GraphNeuralNetworks
using Random 

const M = 7 #num machines
const N = 7#num jobs
const P_MIN = 1#min time
const P_MAX = 5#max time

const N_NODES = M*N
const T = M*N+1 #[nodes, T, S]
const S = M*N+2 


i2mn(i) = CartesianIndices((M,N))[i]

struct GameSpec <: GI.AbstractGameSpec end

function generate_conjuctive_edges(rng::AbstractRNG)
  order = reduce(vcat,[(randperm(rng, M).+i*M) for i in 0:N-1])#flattend order of operations
  target = zeros(UInt8, N_NODES+1+N)
  target[T] = T
  n = 0
  for i in 1:N*M
    if(i%M == 1) #first in row: s->
      target[S+n] = order[i]
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

function generate_done_time(p_time, conj_tar, start_nodes)
  done_time = zeros(UInt16, S)
  for o in start_nodes
    last_done_time = 0
    while(true)#propagate expected done time
      o = conj_tar[o]
      last_done_time = done_time[o] = max(last_done_time + p_time[o], done_time[o])
      o == T && break
    end
  end
  return done_time
end

mutable struct GameEnv <: GI.AbstractGameEnv 
  #Problem instance
  process_time::Vector{UInt8} #Index in Mxn
  conj_src::Vector{UInt8}
  conj_tar::Vector{UInt8}
  UB::UInt16
  LB::UInt16
  #State
  disj_src::Vector{UInt8}
  disj_tar::Vector{UInt8}
  is_done::Vector{Bool}
  done_time::Vector{UInt16}
  #Info
  prev_operation::Vector{UInt8}
  prev_machine::Vector{UInt8}
end

function GI.init(::GameSpec; rng::AbstractRNG = Random.GLOBAL_RNG) 
  p_time = [rand(rng, P_MIN:P_MAX,N_NODES); 0; 0] #Nodes time plus t, s = 0
  conj_tar = generate_conjuctive_edges(rng)
  start_nodes = collect(0:N-1) .+ S
  return GameEnv(
    #Problem instance
    p_time,
    [collect(1:N*M)..., T, S * ones(N)...],
    conj_tar,
    sum(p_time),
    maximum(sum.([p_time[i*M+1:(i+1)*M] for i in 0:N-1])),
    #State
    collect(1:N_NODES),
    collect(1:N_NODES),
    [falses(N*M)..., false, true], #[nodes, T, S]
    generate_done_time(p_time, conj_tar, start_nodes),
    #Info
    start_nodes, 
    S * ones(UInt8, M)
  )
end

GI.init(::GameSpec, s) = GameEnv(
  #Static values
  s.process_time,
  s.conj_src,
  s.conj_tar,
  s.UB,
  s.LB,
  #Mutable values
  copy(s.disj_src),
  copy(s.disj_tar),
  copy(s.is_done),
  copy(s.done_time),
  copy(s.prev_operation),
  copy(s.prev_machine),
)

GI.spec(::GameEnv) = GameSpec()

GI.two_players(::GameSpec) = false

GI.state_dim(::GameSpec) = (2, S)

GI.set_state!(g::GameEnv, s) = g = s

GI.current_state(g::GameEnv) = (
  process_time = g.process_time,
  conj_src = g.conj_src,
  conj_tar = g.conj_tar,
  UB = g.UB,
  LB = g.LB,
  disj_src = copy(g.disj_src),
  disj_tar = copy(g.disj_tar),
  is_done = copy(g.is_done),
  done_time = copy(g.done_time),
  prev_operation = copy(g.prev_operation),
  prev_machine = copy(g.prev_machine),
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
  #update info vectors
  g.prev_machine[mn[1]] = o
  g.prev_operation[mn[2]] = o
  #determine if all opperations are done
  all(g.conj_tar[g.prev_operation].==T) && (g.is_done[T] = true)
  #update done time
  last_done_time = g.done_time[o] = max(g.done_time[l], g.done_time[k] * g.is_done[k]) + g.process_time[o]
  #println("o:", o, " k:", k, " l:", l, " do:", last_done_time*1, " dk:", g.done_time[k] * g.is_done[k]*1, " dl:", g.done_time[l]*1)
  while(true)#propagate expected done time
    o = g.conj_tar[o]
    last_done_time = g.done_time[o] = max(g.done_time[o], last_done_time + g.process_time[o])
    o == T && return
  end
end

function GI.white_reward(g::GameEnv)
  if(all(g.is_done))
    #println("dt:", g.done_time[T], " ub:", g.UB, " lb:", g.LB, " vl:", (g.UB - g.done_time[T])/(g.UB - g.LB))
    return ((g.UB - g.done_time[T])/(g.UB - g.LB))
  end
  return 0
end

function GI.vectorize_state(::GameSpec, state) 
  # println([state.conj_src; state.disj_src])
  # println([state.conj_tar; state.disj_tar])
  # println(S)
  # println(Float32.(hcat(state.done_time, state.is_done)'))
  return  GNNGraph([state.conj_src; state.disj_src], 
                   [state.conj_tar; state.disj_tar], 
                   num_nodes = S, 
                   ndata = Float32.(hcat(state.done_time, state.is_done)'))
end

# function GI.symmetries(::GameSpec, s)

# end

#####
##### Interaction APIc
#####

function GI.action_string(::GameSpec, a)

end

function GI.parse_action(::GameSpec, str)

end


function GI.read_state(::GameSpec)

end


function GI.render(g::GameEnv; with_position_names=true, botmargin=true)

end
