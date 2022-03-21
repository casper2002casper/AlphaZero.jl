import AlphaZero.GI
using GraphNeuralNetworks
using Random 
using Crayons


i2mn(i, M, N) = CartesianIndices((M,N))[i]
mn2i(m, n, M, N) = LinearIndices((M,N))[m,n]

struct GameSpec <: GI.AbstractGameSpec 
  M::UInt8
  N::UInt8
  P_MIN::UInt8
  P_MAX::UInt8
end

function generate_conjuctive_edges(rng::AbstractRNG, M, N, N_NODES, T, S)
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

function generate_done_time(p_time, conj_tar, start_edges, S, T)
  done_time = zeros(UInt16, S)
  for o in start_edges
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
  M::UInt8
  N::UInt8
  N_NODES::UInt16
  T::UInt16
  S::UInt16
  # UB::UInt16
  # LB::UInt16
  #State
  disj_src::Vector{UInt8}
  disj_tar::Vector{UInt8}
  is_done::Vector{Bool}
  done_time::Vector{UInt16}
  #Info
  prev_operation::Vector{UInt8}
  prev_machine::Vector{UInt8}
end

function GI.init(spec::GameSpec, rng::AbstractRNG) 
  M = spec.M
  N = spec.M
  N_NODES = M*N #[nodes, T, S]
  T = M*N+1 
  S = M*N+2 
  p_time = [rand(rng, spec.P_MIN:spec.P_MAX, N_NODES); 0; 0] #Nodes time plus t, s = 0
  conj_tar = generate_conjuctive_edges(rng, M, N, N_NODES, T, S)
  start_edges_n = collect(0:N-1) .+ S
  start_edges_m = collect(0:M-1) .+ S
  return GameEnv(
    #Problem instance
    p_time,
    [collect(1:N_NODES)..., T, S * ones(N)...],
    conj_tar,
    M,
    N,
    N_NODES,
    T,
    S,
    # sum(p_time),
    # maximum(sum.([p_time[i*M+1:(i+1)*M] for i in 0:N-1])),
    #State
    [collect(1:N_NODES)..., T, S * ones(M)...],
    [collect(1:N_NODES)..., T, S * ones(M)...],
    [falses(N_NODES)..., false, true], #[nodes, T, S]
    generate_done_time(p_time, conj_tar, start_edges_n, S, T),
    #Info
    start_edges_n, 
    start_edges_m
  )
end

GI.init(spec::GameSpec, s) = GameEnv(
  #Static values
  s.process_time,
  s.conj_src,
  s.conj_tar,
  s.M,
  s.N,
  s.N_NODES,
  s.T,
  s.S,
  # s.UB,
  # s.LB,
  #Mutable values
  copy(s.disj_src),
  copy(s.disj_tar),
  copy(s.is_done),
  copy(s.done_time),
  copy(s.prev_operation),
  copy(s.prev_machine),
)

GI.spec(g::GameEnv) = GameSpec(g.M, g.N, 1, 5)

GI.two_players(::GameSpec) = false

GI.state_dim(spec::GameSpec) = (2, (spec.M*spec.N+2))#Opperations + source and sink

GI.set_state!(g::GameEnv, s) = g = s

GI.current_state(g::GameEnv) = (
  process_time = g.process_time,
  conj_src = g.conj_src,
  conj_tar = g.conj_tar,
  M = g.M,
  N = g.N,
  N_NODES = g.N_NODES,
  T = g.T,
  S = g.S,
  # UB = g.UB,
  # LB = g.LB,
  disj_src = copy(g.disj_src),
  disj_tar = copy(g.disj_tar),
  is_done = copy(g.is_done),
  done_time = copy(g.done_time),
  prev_operation = copy(g.prev_operation),
  prev_machine = copy(g.prev_machine),
)

GI.white_playing(g::GameEnv) = true

GI.game_terminated(g::GameEnv) = all(g.is_done)

GI.actions(spec::GameSpec) = collect(1:(spec.M*spec.N+2)) 

function GI.actions_mask(g::GameEnv)
  valid_action = zeros(Bool, g.S)
  valid_action[g.conj_tar[g.prev_operation]] .= true
  valid_action[g.T] = false #We cant schedule sink node
  return valid_action
end

function GI.play!(g::GameEnv, o)
  #mark operation scheduled
  g.is_done[o] = true
  mn = i2mn(o, g.M, g.N)
  #update previous operation and machine
  k = g.prev_machine[mn[1]] #previous operation done on machine of todo operation
  l = g.prev_operation[mn[2]] #previous operation done in job of todo operation
  #update info vectors
  g.prev_machine[mn[1]] = o
  g.prev_operation[mn[2]] = o
  #determine if all opperations are done
  all(g.conj_tar[g.prev_operation].==g.T) && (g.is_done[g.T] = true)
  #add disjunctive link
  g.disj_tar[k] = o
  #convert from edge to node notation
  l = min(l, g.S)
  k = min(k, g.S)
  #update done time
  last_done_time = g.done_time[o] = max(g.done_time[l], g.done_time[k] * g.is_done[k]) + g.process_time[o]
  while(true)#propagate expected done time
    o = g.conj_tar[o]
    last_done_time = g.done_time[o] = max(g.done_time[o], last_done_time + g.process_time[o])
    o == g.T && return
  end
end

function GI.white_reward(g::GameEnv)
  if(all(g.is_done))
    #println("dt:", g.done_time[T], " ub:", g.UB, " lb:", g.LB, " vl:", (g.UB - g.done_time[T])/(g.UB - g.LB))
    return -convert(Float32, g.done_time[g.T])
    #return ((g.UB - g.done_time[T])/(g.UB - g.LB))
  end
  return 0
end

function GI.vectorize_state(::GameSpec, state) 
  return  GNNGraph([state.conj_src; state.disj_src], 
                   [state.conj_tar; state.disj_tar], 
                   num_nodes = state.S, 
                   ndata = Float32.(hcat(state.done_time, state.is_done)'))
end

#####
##### Interaction APIc
#####

function GI.action_string(spec::GameSpec, o)
  mn = i2mn(o, spec.M, spec.N)
  return string("job: ", mn[2], " machine: ", mn[1])
end

function GI.parse_action(spec::GameSpec, str)
  try
    s = split(str)
    @assert length(s) == 2
    n = parse(Int, s[1])
    m = parse(Int, s[2])
    return mn2i(m,n, spec.M, spec.N)
  catch e
    return nothing
  end
end


function GI.read_state(::GameSpec)#input problem
  return nothing
end


function GI.render(g::GameEnv) 
  for (m,o) in enumerate(g.disj_tar[g.S:end])
    print(crayon"white", "m", m, ":")
    o == g.S && (println(); continue)
    last_op = 0
    while(true)
      mn = i2mn(o, g.M, g.N)
      idle = (crayon"dark_gray", repeat("-", g.done_time[o] - (last_op + g.process_time[o])))
      active = (Crayon(foreground = mn[2]), repeat("=", g.process_time[o]))
      print(idle..., active...)
      last_op = g.done_time[o]
      g.disj_tar[o] == o && break
      o = g.disj_tar[o]
    end
    println(crayon"reset")
  end
  for n in 1:N
    print(Crayon(foreground = n), "n", n, " ")
  end
  println(crayon"reset")
end
