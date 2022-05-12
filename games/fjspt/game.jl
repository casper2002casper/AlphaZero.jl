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
  K::Pair{AbstractSchedule,AbstractSchedule}
  P::Pair{UInt8,UInt8}
  T::Pair{UInt8,UInt8}
end

mutable struct AdaptiveNodes
  src::Matrix{UInt8}
  tar::Matrix{UInt8}
  info::Matrix{UInt8}
  done_time::Matrix{UInt16}
end

Base.copy(s::AdaptiveNodes) = AdaptiveNodes(copy(s.src), copy(s.tar), copy(s.info), copy(s.done_time))

function generate_conjuctive_edges(rng::AbstractRNG, M, N, N_OPP, T, S)
  order = reduce(vcat, [(randperm(rng, M) .+ i * M) for i in 0:N-1]) * 2 .-1#order of operations
  target = zeros(UInt8, N_OPP*2 + 1 + N)
  target[T] = T
  n = 0
  for i in 1:N*M
    if (i % M == 1) #first in row: s->
      target[S + n] = order[i]
    end
    target[order[i]] = order[i]+1#link to m_node
    if (i % M == 0)#last in row: ->t
      target[order[i]+1] = T
      n += 1
    else #propagate
      target[order[i]+1]  = order[i+1]#to next t_node
    end
  end
  return target
end

function gen_done_time(p_time, t_time, conj_tar, start_edges, S, T)
  done_time = zeros(UInt16, S)
  for o in start_edges
    last_done_time = 0
    last_m = size(t_time,1)#M+2
    while (true)#propagate expected done time
      t = conj_tar[o]
      o = conj_tar[t]
      if (o == T)
        done_time[T] = max(done_time[T], last_done_time)
        break
      end
      m_time, m = findmin(p_time[o÷2,:])
      done_time[t] = last_done_time + t_time[last_m,m]
      last_done_time = done_time[o] = done_time[t] + m_time
      last_m = m
    end
  end
  return done_time
end

function gen_action_values(p_time, t_time, conj_tar, start_edges, K, S)
  nodes = AdaptiveNodes(Matrix{UInt8}(undef, 0, 5), Matrix{UInt8}(undef, 0, 5), Matrix{UInt8}(undef, 0, 3),  Matrix{UInt8}(undef, 0, 2))
  for e in start_edges
    t = conj_tar[e]
    o = conj_tar[t]
    for (m, m_time) in enumerate(p_time[o÷2,:])
      p_time==0xff && continue  
      for k in 1:K  
        t_node_id = S + length(nodes.done_time) + 1
        m_node_id = t_node_id + 1
        nodes.src = [nodes.src; [S S t_node_id m_node_id S]]
        nodes.tar = [nodes.tar; [t_node_id t_node_id m_node_id conj_tar[o] m_node_id]]
        nodes.info = [nodes.info; [m k o]]
        nodes.done_time = [nodes.done_time; [t_time[end,m] t_time[end,m]+m_time]] 
      end
    end
  end
  return nodes
end

mutable struct GameEnv <: GI.AbstractGameEnv
  #Problem instance
  process_time::Matrix{UInt8} #Index in Mxn
  transport_time::Matrix{UInt8} #M+2xM+2
  M::UInt8
  N::UInt8
  K::UInt8
  N_OPP::UInt16
  T::UInt16
  S::UInt16
  UB::UInt16
  LB::UInt16
  conj_src::Vector{UInt8}
  conj_tar::Vector{UInt8}
  #State
  disj_src::Vector{UInt8}
  disj_tar::Vector{UInt8}
  is_done::Vector{Bool}
  done_time::Vector{UInt16}
  adaptive_nodes::AdaptiveNodes
  #Info
  prev_operation::Vector{UInt8}
  prev_machine::Matrix{UInt8}
  prev_vehicle::Matrix{UInt8}
end

function GI.init(spec::GameSpec, itc::Int, rng::AbstractRNG)
  N = rand(rng, spec.N.first[itc]:spec.N.second[itc])
  M = rand(rng, spec.M.first[itc]:spec.M.second[itc])
  K = rand(rng, spec.K.first[itc]:spec.K.second[itc])
  N_OPP = N * M #[T_nodes, M_nodes, T, S]
  T = N_OPP * 2 + 1
  S = N_OPP * 2 + 2
  p_time = rand(rng, spec.P.first:spec.P.second, N_OPP, M) 
  for o in 1:N_OPP
    ind = randperm(M)[1:M-rand(rng, spec.A.first[itc]:spec.A.second[itc])]
    p_time[o,ind] .= 0xff
  end
  t_time = rand(rng, spec.T.first:spec.T.second, M + 2, M + 2)
  conj_tar = generate_conjuctive_edges(rng, M, N, N_OPP, T, S)
  start_edges_n = collect(0:N-1) .+ S
  start_edges_m = collect(0:M-1) .+ S
  node_done = gen_done_time(p_time, t_time, conj_tar, start_edges_n, S, T)
  adaptive_nodes = gen_action_values(p_time, t_time, conj_tar, start_edges_n, K, S)
  return GameEnv(
    #Problem instance
    p_time,
    t_time,
    M,
    N,
    K,
    N_OPP,
    T,
    S,
    sum(maximum.(values.(p_time))), #All worst operations in a row
    maximum(sum.([minimum.(values.(p_time[i*M+1:(i+1)*M])) for i in 0:N-1])),#Best operations parrallel
    [collect(1:N_OPP*2); T; S * ones(N)],
    conj_tar,
    #State
    [],
    [],
    [falses(N_OPP*2); false; true], #[Transport, Operations, T, S]
    node_done,
    adaptive_nodes,
    #Info
    start_edges_n,
    [start_edges_m repeat([S], M)],
    ones(UInt8, K, 2) * S
  )
end

GI.init(spec::GameSpec, s) = GameEnv(
  #Static values
  s.process_time,
  s.transport_time,
  s.M,
  s.N,
  s.K,
  s.N_OPP,
  s.T,
  s.S,
  s.UB,
  s.LB,
  s.conj_src,
  s.conj_tar,
  #Mutable values
  copy(s.disj_src),
  copy(s.disj_tar),
  copy(s.is_done),
  copy(s.done_time),
  copy(s.adaptive_nodes),
  copy(s.prev_operation),
  copy(s.prev_machine),
  copy(s.prev_vehicle)
)

GI.spec(g::GameEnv) = GameSpec(ConstSchedule(g.M) => ConstSchedule(g.M), ConstSchedule(g.N) => ConstSchedule(g.N), ConstSchedule(2) => ConstSchedule(2), ConstSchedule(2) => ConstSchedule(2), 1 => 5, 1 => 5)

GI.two_players(::GameSpec) = false

GI.state_dim(spec::GameSpec) = (2, (spec.M.second[1] * spec.N.second[1] + 2))#Opperations + source and sink

function GI.set_state!(g::GameEnv, s)
  g.process_time = s.process_time
  g.transport_time = s.transport_time
  g.M = s.M
  g.N = s.N
  g.K = s.K
  g.N_OPP = s.N_OPP
  g.T = s.T
  g.S = s.S
  g.UB = s.UB
  g.LB = s.LB
  g.conj_src = s.conj_src
  g.conj_tar = s.conj_tar
  g.disj_src = copy(s.disj_src)
  g.disj_tar = copy(s.disj_tar)
  g.is_done = copy(s.is_done)
  g.done_time = copy(s.done_time)
  g.adaptive_nodes = copy(s.adaptive_nodes)
  g.prev_operation = copy(s.prev_operation)
  g.prev_machine = copy(s.prev_machine)
  g.prev_vehicle = copy(s.prev_vehicle)
end

GI.current_state(g::GameEnv) = (
  process_time=g.process_time,
  transport_time=g.transport_time,
  M=g.M,
  N=g.N,
  K=g.K,
  N_OPP=g.N_OPP,
  T=g.T,
  S=g.S,
  UB=g.UB,
  LB=g.LB,
  conj_src=g.conj_src,
  conj_tar=g.conj_tar,
  disj_src=copy(g.disj_src),
  disj_tar=copy(g.disj_tar),
  is_done=copy(g.is_done),
  done_time=copy(g.done_time),
  adaptive_nodes=copy(g.adaptive_nodes),
  prev_operation=copy(g.prev_operation),
  prev_machine=copy(g.prev_machine),
  prev_vehicle=copy(g.prev_vehicle)
)

GI.white_playing(g::GameEnv) = true

GI.game_terminated(g::GameEnv) = g.is_done[g.T]

GI.actions(spec::GameSpec) = collect(1:(spec.M.second[1]*spec.N.second[1]+2))

GI.available_actions(g::GameEnv) = collect(g.S+2:2:g.S+length(g.adaptive_nodes.done_time))

GI.actions_mask(g::GameEnv) = [falses(g.S); repeat([false, true], size(g.adaptive_nodes.done_time,1))]#fix


function GI.play!(g::GameEnv, action)
  #mark operation scheduled
  m, k, o = g.adaptive_nodes.info[(action-g.S)÷2, :]
  t = o - g.N_OPP
  @assert g.is_done[o] == false
  g.is_done[o] = true
  g.is_done[t] = true
  i = o2ji(o, g.M, g.N)[2]
  #update previous operation and machine
  p = g.prev_machine[m] #previous operation done on machine of todo operation
  l, m_l = g.prev_operation[i,:] #previous operation done in job of todo operation
  u, m_u = g.prev_vehicle[k,:]
  #update info vectors
  g.prev_machine[m] = o
  g.prev_operation[i,:] = [o, m]
  g.prev_vehicle[k,:] = [t, m]
  #add disjunctive link
  append!(g.disj_src, [u p])
  append!(g.disj_tar, [t o])
  #convert from edge to node notation
  l = min(l, g.S)
  p = min(p, g.S)
  #set done time
  g.done_time[t] = max(g.done_time[l], g.done_time[u] + g.transport_time[m_u, m_l]) + g.transport_time[m_l, m] 
  g.done_time[o] = max(g.done_time[o, 1], g.done_time[p]) + get(g.process_time[o], m, nothing)
  #remove old actions
  mask = g.nodes.info[:, 3] .!== o
  g.adaptive_nodes.src = g.adaptive_nodes.src[mask, :]
  g.adaptive_nodes.tar = g.adaptive_nodes.tar[mask, :]
  g.adaptive_nodes.info = g.adaptive_nodes.info[mask, :]
  g.adaptive_nodes.done_time = g.adaptive_nodes.done_time[mask,:]
  #fix disjunctive edges 
  g.adaptive_nodes.src[g.adaptive_nodes.info[:,2].==k, 2] .= t
  g.adaptive_nodes.src[g.adaptive_nodes.info[:,1].==m, 5] .= o #fix
  #fix node ids
  num_adaptive_nodes = size(g.adaptive_nodes.done_time,1)*2
  t_ids = collect(g.S+1:2:g.S+num_adaptive_nodes)
  m_ids = t_ids.+1
  g.adaptive_nodes.tar[:, 1] = t_ids
  g.adaptive_nodes.tar[:, 2] = t_ids
  g.adaptive_nodes.src[:, 3] = t_ids
  g.adaptive_nodes.src[:, 4] = m_ids
  g.adaptive_nodes.tar[:, 3] = m_ids
  g.adaptive_nodes.tar[:, 5] = m_ids
  #add new actions
  next_t = g.conj_tar[o]
  next_o = g.conj_tar[next_t]
  next_next_t = g.conj_tar[next_o]
  next_next_o= g.conj_tar[next_next_t]
  for (m_, p_time) in enumerate(p_time[o÷2,:])
    p_time==0xff && continue  
    for k in 1:g.K
      t_node_id = S + size(g.nodes.done_time,1)*2 + 1
      m_node_id = t_node_id + 1
      g.adaptive_nodes.src = [g.adaptive_nodes.src; [o g.prev_vehicle[k] t_node_id m_node_id g.prev_machine[m_]]]
      g.adaptive_nodes.tar = [g.adaptive_nodes.tar; [t_node_id t_node_id m_node_id next_next_t m_node_id]]
      g.adaptive_nodes.info = [g.adaptive_nodes.info; [m k next_next_o]]
      transport_done = g.done_time[o]+g.transport_time[m, m_]
      g.adaptive_nodes.done_time = [g.adaptive_nodes.done_time; [transport_done transport_done+p_time]]
    end
  end
  #mark last node as done
  size(g.action_done_time, 1) == 0 && (g.is_done[g.T] = true)
  #propagate expected done time
  last_done_time = g.done_time[o]
  last_m = m
  while (true)
    t = g.conj_tar[o]
    o = g.conj_tar[t]
    if (o == g.T)
      g.done_time[g.T] = max(g.done_time[g.T], last_done_time)
      return
    end
    m_time, m = findmin(p_time[o÷2,:])
    done_time[t] = last_done_time + t_time[last_m,m]
    last_done_time = done_time[o] = done_time[t] + m_time
    last_m = m
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
  num_actions = length(state.adaptive_nodes.done_time)
  return GNNGraph([state.conj_src; state.disj_src; vec(state.adaptive_nodes.src)],
    [state.conj_tar; state.disj_tar; vec(state.adaptive_nodes.tar)],
    num_nodes=state.S + num_actions,
    ndata=Float32.([[state.done_time; vec(state.adaptive_nodes.done_time)] [state.is_done; zeros(num_actions)]]'))
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
