import AlphaZero.GI
using AlphaZero: AbstractSchedule, ConstSchedule
using GraphNeuralNetworks
using Random
using Crayons

o2ji(o, M, N) = CartesianIndices((M, N))[o]
ji2o(j, i, M, N) = LinearIndices((M, N))[j, i]

struct GameSpec <: GI.AbstractGameSpec
  M::Pair{AbstractSchedule,AbstractSchedule}
  N::Pair{AbstractSchedule,AbstractSchedule}
  A::Pair{AbstractSchedule,AbstractSchedule}
  P::Pair{UInt8,UInt8}
end

function generate_conjuctive_edges(rng::AbstractRNG, M, N, N_OPP, T, S)
  order = reduce(vcat, [(randperm(rng, M) .+ i * M) for i in 0:N-1])#order of operations
  target = zeros(UInt8, N_OPP + 1 + N)
  target[T] = T
  n = 0
  for i in 1:N*M
    if (i % M == 1) #first in row: s->
      target[S+n] = order[i]
    end
    if (i % M == 0)#last in row: ->t
      target[order[i]] = T
      n += 1
    else #propagate
      target[order[i]] = order[i+1]
    end
  end
  return target
end

function gen_done_time(p_time, conj_tar, start_edges, S, T)
  done_time = zeros(UInt16, S)
  for o in start_edges
    last_done_time = 0
    while (true)#propagate expected done time
      o = conj_tar[o]
      if (o == T)
        done_time[T] = max(done_time[T], last_done_time)
        break
      end
      last_done_time = done_time[o] = last_done_time + minimum(values(p_time[o]))
    end
  end
  return done_time
end

function gen_action_values(p_time, conj_tar, start_edges, S, T)
  src = Matrix{UInt8}(undef, 0, 3)
  tar = Matrix{UInt8}(undef, 0, 3)
  info = Matrix{UInt8}(undef, 0, 2)
  done_time = []
  for (i, e) in enumerate(start_edges)
    o = conj_tar[e]
    for (m, time) in p_time[o]
      self = length(done_time) + S + 1
      src = [src; [S self S]] 
      tar = [tar; [self conj_tar[o] self]]
      info = [info; [m o]]
      done_time = [done_time; time]
    end
  end
  return src, tar, info, done_time
end

mutable struct GameEnv <: GI.AbstractGameEnv
  #Problem instance
  process_time::Vector{Dict{UInt8,UInt8}} #Index in Mxn
  M::UInt8
  N::UInt8
  N_OPP::UInt16
  T::UInt16
  S::UInt16
  UB::UInt16
  LB::UInt16
  conj_src::Vector{UInt8}
  conj_tar::Vector{UInt8}
  #State
  action_src::Matrix{UInt8}
  action_tar::Matrix{UInt8}
  action_info::Matrix{UInt8}
  disj_src::Vector{UInt8}
  disj_tar::Vector{UInt8}
  is_done::Vector{Bool}
  done_time::Vector{UInt16}
  action_done_time::Vector{UInt16}
  #Info
  prev_operation::Vector{UInt8}
  prev_machine::Vector{UInt8}
end

function GI.init(spec::GameSpec, itc::Int, rng::AbstractRNG)
  M = rand(rng, spec.M.first[itc]:spec.M.second[itc])
  N = rand(rng, spec.N.first[itc]:spec.N.second[itc])
  N_OPP = M * N #[nodes, T, S]
  T = M * N + 1
  S = M * N + 2
  gen_node() = Dict([UInt8(i) => rand(rng, spec.P.first:spec.P.second) for i in randperm(M)[1:rand(rng, spec.A.first[itc]:spec.A.second[itc])]])
  p_time = [gen_node() for _ in 1:N_OPP]
  conj_tar = generate_conjuctive_edges(rng, M, N, N_OPP, T, S)
  start_edges_n = collect(0:N-1) .+ S
  start_edges_m = collect(0:M-1) .+ S
  node_done = gen_done_time(p_time, conj_tar, start_edges_n, S, T)
  action_src, action_tar, action_info, action_done = gen_action_values(p_time, conj_tar, start_edges_n, S, T)
  return GameEnv(
    #Problem instance
    p_time,
    M,
    N,
    N_OPP,
    T,
    S,
    sum(maximum.(values.(p_time))), #All worst operations in a row
    maximum(sum.([minimum.(values.(p_time[i*M+1:(i+1)*M])) for i in 0:N-1])),#Best operations parrallel
    [collect(1:N_OPP); T; S * ones(N)],
    conj_tar,
    #State
    action_src,
    action_tar,
    action_info,
    [],
    [],
    [falses(N_OPP); false; true], #[Operations, T, S]
    node_done,
    action_done,
    #Info
    start_edges_n,
    start_edges_m
  )
end

GI.init(spec::GameSpec, s) = GameEnv(
  #Static values
  s.process_time,
  s.M,
  s.N,
  s.N_OPP,
  s.T,
  s.S,
  s.UB,
  s.LB,
  s.conj_src,
  s.conj_tar,
  #Mutable values
  copy(s.action_src),
  copy(s.action_tar),
  copy(s.action_info),
  copy(s.disj_src),
  copy(s.disj_tar),
  copy(s.is_done),
  copy(s.done_time),
  copy(s.action_done_time),
  copy(s.prev_operation),
  copy(s.prev_machine),
)

GI.spec(g::GameEnv) = GameSpec(ConstSchedule(g.M) => ConstSchedule(g.M), ConstSchedule(g.N) => ConstSchedule(g.N), ConstSchedule(3) => ConstSchedule(3), 1 => 5)

GI.two_players(::GameSpec) = false

GI.state_dim(spec::GameSpec) = (2, (spec.M.second[1] * spec.N.second[1] + 2))#Opperations + source and sink

function GI.set_state!(g::GameEnv, s)
  g.process_time = s.process_time
  g.M = s.M
  g.N = s.N
  g.N_OPP = s.N_OPP
  g.T = s.T
  g.S = s.S
  g.UB = s.UB
  g.LB = s.LB
  g.conj_src = s.conj_src
  g.conj_tar = s.conj_tar
  g.action_src = copy(s.action_src)
  g.action_tar = copy(s.action_tar)
  g.action_info = copy(s.action_info)
  g.disj_src = copy(s.disj_src)
  g.disj_tar = copy(s.disj_tar)
  g.is_done = copy(s.is_done)
  g.done_time = copy(s.done_time)
  g.action_done_time = copy(s.action_done_time)
  g.prev_operation = copy(s.prev_operation)
  g.prev_machine = copy(s.prev_machine)
end

GI.current_state(g::GameEnv) = (
  process_time=g.process_time,
  M=g.M,
  N=g.N,
  N_OPP=g.N_OPP,
  T=g.T,
  S=g.S,
  UB=g.UB,
  LB=g.LB,
  conj_src=g.conj_src,
  conj_tar=g.conj_tar,
  action_src=copy(g.action_src),
  action_tar=copy(g.action_tar),
  action_info=copy(g.action_info),
  disj_src=copy(g.disj_src),
  disj_tar=copy(g.disj_tar),
  is_done=copy(g.is_done),
  done_time=copy(g.done_time),
  action_done_time=copy(g.action_done_time),
  prev_operation=copy(g.prev_operation),
  prev_machine=copy(g.prev_machine),
)

GI.white_playing(g::GameEnv) = true

GI.game_terminated(g::GameEnv) = g.is_done[g.T]

GI.actions(spec::GameSpec) = collect(1:(spec.M.second[1]*spec.N.second[1]+2))

GI.available_actions(g::GameEnv) = collect(g.S+1:g.S+length(g.action_done_time)+1)

GI.actions_mask(g::GameEnv) = [falses(g.S); trues(length(g.action_done_time))]

function GI.play!(g::GameEnv, action)
  #mark operation scheduled
  m, o = g.action_info[action-g.S, :]
  @assert g.is_done[o] == false
  g.is_done[o] = true
  i = o2ji(o, g.M, g.N)[2]
  #update previous operation and machine
  k = g.prev_machine[m] #previous operation done on machine of todo operation
  l = g.prev_operation[i] #previous operation done in job of todo operation
  #update info vectors
  g.prev_machine[m] = o
  g.prev_operation[i] = o
  #add disjunctive link
  append!(g.disj_src, k) 
  append!(g.disj_tar, o)
  #convert from edge to node notation
  l = min(l, g.S)
  k = min(k, g.S)
  g.done_time[o] = max(g.done_time[l], g.done_time[k] * g.is_done[k]) + get(g.process_time[o], m, nothing)
  #remove old actions
  mask = g.action_info[:, 2] .!== o
  g.action_src = g.action_src[mask, :]
  g.action_tar = g.action_tar[mask, :]
  g.action_info = g.action_info[mask, :]
  g.action_done_time = g.action_done_time[mask]
  #fix disjunctive edges 
  g.action_src[g.action_info[:, 1] .== m, 3] .= o
  #fix self reference
  g.action_src[:, 2] = collect(g.S+1:g.S+length(g.action_done_time))
  g.action_tar[:, 1] = g.action_src[:, 2]
  g.action_tar[:, 3] = g.action_src[:, 2]
  #add new actions
  next_o = g.conj_tar[o]
  next_next_o = g.conj_tar[next_o]
  for (m, p_time) in get(g.process_time, next_o, [])
    self = g.S + length(g.action_done_time) + 1
    g.action_src = [g.action_src; [o self min(g.prev_machine[m], g.S)]]
    g.action_tar = [g.action_tar; [self next_next_o self]]
    g.action_info = [g.action_info; [m next_o]]
    append!(g.action_done_time, g.done_time[o] + p_time)
  end
  #mark last operation as done
  length(g.action_done_time)==0 && (g.is_done[g.T] = true)
  #propagate expected done time
  last_done_time = g.done_time[o]
  while (true)
    o = g.conj_tar[o]
    if (o == g.T)
      g.done_time[g.T] = max(g.done_time[g.T], last_done_time)
      return
    end
    last_done_time = g.done_time[o] = last_done_time + minimum(values(g.process_time[o]))
  end
end

function GI.white_reward(g::GameEnv)
  if (all(g.is_done))
    #return -convert(Float32, g.done_time[g.T])
    return ((g.UB - g.done_time[g.T]) / (g.UB - g.LB))
  end
  return 0
end

function GI.vectorize_state(::GameSpec, state)
  num_actions = length(state.action_done_time)
  return GNNGraph([state.conj_src; state.disj_src; vec(state.action_src)],
    [state.conj_tar; state.disj_tar; vec(state.action_tar)],
    num_nodes = state.S + num_actions,
    ndata=Float32.([[state.done_time; state.action_done_time] [state.is_done; zeros(num_actions)]]'))
end

#####
##### Interaction APIc
#####

function GI.action_string(spec::GameSpec, o)
  mn = o2ji(o, spec.M.second, spec.N.second)
  return string("job: ", mn[2], " machine: ", mn[1])
end

function GI.parse_action(spec::GameSpec, str)
  try
    s = split(str)
    @assert length(s) == 2
    n = parse(Int, s[1])
    m = parse(Int, s[2])
    return ji2o(m, n, spec.M.second, spec.N.second)
  catch e
    return nothing
  end
end


function GI.read_state(::GameSpec)#input problem
  return nothing
end


function GI.render(g::GameEnv)
  for (m, o) in enumerate(g.disj_tar[g.S:end])
    print(crayon"white", "m", m, ":")
    o == g.S && (println(); continue)
    last_op = 0
    while (true)
      mn = o2ji(o, g.M, g.N)
      idle = (crayon"dark_gray", repeat("-", g.done_time[o] - (last_op + g.process_time[o])))
      active = (Crayon(foreground=mn[2]), repeat("=", g.process_time[o]))
      print(idle..., active...)
      last_op = g.done_time[o]
      g.disj_tar[o] == o && break
      o = g.disj_tar[o]
    end
    println(crayon"reset")
  end
  for n in 1:g.N
    print(Crayon(foreground=n), "n", n, " ")
  end
  println(crayon"reset")
end
