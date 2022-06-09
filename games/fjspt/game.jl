import AlphaZero.GI
import Cairo, Fontconfig
using AlphaZero: AbstractSchedule, ConstSchedule
using GraphNeuralNetworks
using Random
using Crayons
using JSON3
using StructTypes
using GraphPlot, Compose, Graphs, Colors

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

function generate_conjuctive_edges(num_operations, N_OPP, T, S)
  N = length(num_operations)
  target = [collect(UInt8, 2:N_OPP*2+1); repeat([T], N + 1); zeros(UInt8, N)]
  index = 1
  for (i, num) in enumerate(num_operations)
    target[S+i-1] = index
    index += num * 2
    target[index-1] = T - i
  end
  return target
end

function gen_done_time(p_time, t_time, conj_tar, start_edges, S, T)
  done_time = zeros(UInt16, S)
  for o in start_edges
    last_done_time = 0
    last_m = size(t_time, 1)#M+2
    while (true)#propagate expected done time
      t = conj_tar[o]
      o = conj_tar[t]
      if (o == T)
        done_time[t] = last_done_time + t_time[last_m, end]
        done_time[T] = max(done_time[T], done_time[t])
        break
      end
      m_time, m = findmin(p_time[o÷2, :])
      done_time[t] = last_done_time + t_time[last_m, m]
      last_done_time = done_time[o] = done_time[t] + m_time
      last_m = m
    end
  end
  return done_time
end

function gen_action_values(p_time, t_time, conj_tar, start_edges, K, S)
  nodes = AdaptiveNodes(Matrix{UInt8}(undef, 0, 5), Matrix{UInt8}(undef, 0, 5), Matrix{UInt8}(undef, 0, 4), Matrix{UInt8}(undef, 0, 2))
  for (i, e) in enumerate(start_edges)
    t = conj_tar[e]
    o = conj_tar[t]
    for (m, m_time) in enumerate(p_time[o÷2, :])
      m_time == 0xff && continue
      for k in 1:K
        t_node_id = UInt8(S + length(nodes.done_time) + 1)
        m_node_id = t_node_id + 0x1
        nodes.src = [nodes.src; [S S t_node_id m_node_id S]]
        nodes.tar = [nodes.tar; [t_node_id t_node_id m_node_id conj_tar[o] m_node_id]]
        nodes.info = [nodes.info; [m k o i]]
        nodes.done_time = [nodes.done_time; [t_time[end, m] t_time[end, m] + m_time]]
      end
    end
  end
  return nodes
end

mutable struct GameEnv <: GI.AbstractGameEnv
  #Problem instance
  process_time::Matrix{UInt8} #Index in (Mxn)*M
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
  prev_operation::Matrix{UInt8}
  prev_machine::Vector{UInt8}
  prev_vehicle::Matrix{UInt8}
  start_machine::Vector{UInt8}
  start_vehicle::Vector{UInt8}
end

function Base.hash(s::GameEnv, h::UInt)
  return hash([s.disj_src; s.disj_tar], h)
end
Base.isequal(a::GameEnv, b::GameEnv) = isequal((a.disj_src, a.disj_tar), (b.disj_src, b.disj_tar))

Base.copy(s::GameEnv) = GameEnv(
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
  copy(s.prev_vehicle),
  copy(s.start_machine),
  copy(s.start_vehicle)
)

function GI.init(spec::GameSpec, itc::Int, rng::AbstractRNG)
  N = rand(rng, spec.N.first[itc]:spec.N.second[itc])
  M = rand(rng, spec.M.first[itc]:spec.M.second[itc])
  K = rand(rng, spec.K.first[itc]:spec.K.second[itc])
  num_operations = [rand(rng, 1:M) for _ in 1:N]
  N_OPP = sum(num_operations)
  T = N_OPP * 2 + N + 1
  S = N_OPP * 2 + N + 2
  p_time = rand(rng, spec.P.first:spec.P.second, N_OPP, M)
  for o in 1:N_OPP
    ind = randperm(rng, M)[1:M-rand(rng, spec.A.first[itc]:spec.A.second[itc])]
    p_time[o, ind] .= 0xff
  end
  t_time = rand(rng, spec.T.first:spec.T.second, M + 1, M + 1)
  for m in 1:M+1
    t_time[m, m] = 0x00
  end
  conj_tar = generate_conjuctive_edges(num_operations, N_OPP, T, S) #fix
  start_edges_n = collect(0:N-1) .+ S
  node_done = gen_done_time(p_time, t_time, conj_tar, start_edges_n, S, T) #fix
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
    sum(maximum(replace(p_time, 0xff => 0x00), dims=2)) + maximum(t_time) * (2 * N_OPP + N), #All worst operations in a row,
    node_done[T],
    [collect(1:N_OPP*2+N); T; S * ones(N)],
    conj_tar,
    #State
    [],
    [],
    [falses(N_OPP * 2 + N); false; true], #[Transport, Operations, returns, T, S]
    node_done,
    adaptive_nodes,
    #Info
    repeat([S (M + 1)], N),
    repeat([S], M),
    repeat([S (M + 1)], K),
    zeros(UInt8, M),
    zeros(UInt8, K)
  )
end

struct FJSPTInstance
  num_vehicles::Int
  num_machines::Int
  num_operations::Vector{Int}
  process_time::Vector{Int}
  transport_time::Vector{Int}
end

StructTypes.StructType(::Type{FJSPTInstance}) = StructTypes.Struct()

function GI.init(spec::GameSpec, instance_string::String)
  instance = JSON3.read(instance_string, FJSPTInstance)
  p_time = reshape(instance.process_time, :, instance.num_machines)
  t_time = reshape(instance.transport_time, :, instance.num_machines + 1)
  t_time = [t_time[2:end, 2:end] t_time[2:end, 1]; t_time[1, 2:end]... 0]#reshape so LU is last
  N = length(instance.num_operations)
  M = instance.num_machines
  K = instance.num_vehicles
  N_OPP = sum(instance.num_operations)
  T = N_OPP * 2 + N + 1
  S = N_OPP * 2 + N + 2
  conj_tar = generate_conjuctive_edges(instance.num_operations, N_OPP, T, S)
  start_edges_n = collect(0:N-1) .+ S
  node_done = gen_done_time(p_time, t_time, conj_tar, start_edges_n, S, T)
  adaptive_nodes = gen_action_values(p_time, t_time, conj_tar, start_edges_n, K, S)
  return GameEnv(
    p_time,
    t_time,
    instance.num_machines,
    N,
    instance.num_vehicles,
    N_OPP,
    T,
    S,
    sum(maximum(replace(p_time, 0xff => 0x00), dims=2)) + maximum(t_time) * (2 * N_OPP + N), #All worst operations in a row,
    node_done[T],
    [collect(1:N_OPP*2+N); T; S * ones(N)],
    conj_tar,
    #State
    [],
    [],
    [falses(N_OPP * 2 + N); false; true],
    node_done,
    adaptive_nodes,
    #Info
    repeat([S (M + 1)], N),
    repeat([S], M),
    repeat([S (M + 1)], K),
    zeros(UInt8, M),
    zeros(UInt8, K)
  )
end

GI.init(spec::GameSpec, s) = copy(s)

GI.spec(g::GameEnv) = GameSpec(ConstSchedule(g.M) => ConstSchedule(g.M), ConstSchedule(g.N) => ConstSchedule(g.N), ConstSchedule(2) => ConstSchedule(2), ConstSchedule(2) => ConstSchedule(2), 1 => 5, 1 => 5)

GI.two_players(::GameSpec) = false

GI.state_dim(spec::GameSpec) = (6, (spec.M.second[1] * spec.N.second[1] + 2))#Opperations + source and sink

function GI.set_state!(g::GameEnv, s)
  g.disj_src = copy(s.disj_src)
  g.disj_tar = copy(s.disj_tar)
  g.is_done = copy(s.is_done)
  g.done_time = copy(s.done_time)
  g.adaptive_nodes = copy(s.adaptive_nodes)
  g.prev_operation = copy(s.prev_operation)
  g.prev_machine = copy(s.prev_machine)
  g.prev_vehicle = copy(s.prev_vehicle)
  g.start_machine = copy(s.start_machine)
  g.start_vehicle = copy(s.start_vehicle)
end

GI.current_state(g::GameEnv) = copy(g)

GI.white_playing(g::GameEnv) = true

GI.game_terminated(g::GameEnv) = g.is_done[g.T]

GI.actions(spec::GameSpec) = collect(1:(spec.M.second[1]*spec.N.second[1]+2))

GI.available_actions(g::GameEnv) = collect(g.S+2:2:g.S+length(g.adaptive_nodes.done_time))

GI.actions_mask(g::GameEnv) = [falses(g.S); repeat([false, true], size(g.adaptive_nodes.done_time, 1))]#fix


function GI.play!(g::GameEnv, action)
  #mark operation scheduled
  m, k, o, i = g.adaptive_nodes.info[(action-g.S)÷2, :]
  t = o - 0x1
  @assert g.is_done[o] == false
  g.is_done[o] = true
  g.is_done[t] = true
  #update previous operation and machine
  p = g.prev_machine[m] #previous operation done on machine of todo operation
  l, m_l = g.prev_operation[i, :] #previous operation done in job of todo operation
  u, m_u = g.prev_vehicle[k, :]
  #update info vectors
  g.prev_machine[m] = o
  g.prev_operation[i, :] = [o, m]
  isequal(m_l, m) || (g.prev_vehicle[k, :] = [t, m])#don't update if no transport needed
  iszero(g.start_machine[m]) && (g.start_machine[m] = o)
  iszero(g.start_vehicle[k]) && (g.start_vehicle[k] = t)
  #add disjunctive link
  append!(g.disj_src, u)
  append!(g.disj_tar, t)
  append!(g.disj_src, p)
  append!(g.disj_tar, o)
  #set done time
  g.done_time[t] = max(g.done_time[p], max(g.done_time[l], g.done_time[u] + g.transport_time[m_u, m_l]) + g.transport_time[m_l, m])
  g.done_time[o] = max(g.done_time[t], g.done_time[p]) + g.process_time[o÷2, m]
  #remove old actions
  mask = g.adaptive_nodes.info[:, 3] .!== o
  g.adaptive_nodes.src = g.adaptive_nodes.src[mask, :]
  g.adaptive_nodes.tar = g.adaptive_nodes.tar[mask, :]
  g.adaptive_nodes.info = g.adaptive_nodes.info[mask, :]
  g.adaptive_nodes.done_time = g.adaptive_nodes.done_time[mask, :]
  #fix disjunctive edges 
  g.adaptive_nodes.src[g.adaptive_nodes.info[:, 2].==k, 2] .= t
  g.adaptive_nodes.src[g.adaptive_nodes.info[:, 1].==m, 5] .= o
  #fix node ids
  num_adaptive_nodes = size(g.adaptive_nodes.done_time, 1) * 2
  t_ids = collect(UInt8, g.S+1:2:g.S+num_adaptive_nodes)
  m_ids = t_ids .+ 0x1
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
  if (!isequal(next_o, g.T))
    for (p_m, p_time) in enumerate(g.process_time[next_o÷2, :])
      p_time == 0xff && continue
      for k in 1:g.K
        t_node_id = UInt8(g.S + length(g.adaptive_nodes.done_time) + 1)
        m_node_id = UInt8(t_node_id + 1)
        g.adaptive_nodes.src = [g.adaptive_nodes.src; [o g.prev_vehicle[k] t_node_id m_node_id g.prev_machine[p_m]]]
        g.adaptive_nodes.tar = [g.adaptive_nodes.tar; [t_node_id t_node_id m_node_id next_next_t m_node_id]]
        g.adaptive_nodes.info = [g.adaptive_nodes.info; [p_m k next_o i]]
        transport_done = g.done_time[o] + g.transport_time[m, p_m]
        g.adaptive_nodes.done_time = [g.adaptive_nodes.done_time; [transport_done transport_done + p_time]]
      end
    end
  else
    g.is_done[next_t] = true
    append!(g.disj_src, t)
    append!(g.disj_tar, next_t)
    g.prev_vehicle[k, :] = [next_t, g.M + 1]
  end
  #mark last node as done
  isempty(g.adaptive_nodes.done_time) && (g.is_done[g.T] = true)
  #propagate expected done time
  last_done_time = g.done_time[o]
  last_m = m
  while (true)
    t = g.conj_tar[o]
    o = g.conj_tar[t]
    if (o == g.T)
      g.done_time[t] = last_done_time + g.transport_time[last_m, end]
      g.done_time[o] = max(g.done_time[o], g.done_time[t])
      return
    end
    m_time, m = findmin(g.process_time[o÷2, :])
    g.done_time[t] = last_done_time + g.transport_time[last_m, m]
    last_done_time = g.done_time[o] = g.done_time[t] + m_time
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
  done_time = [state.done_time; vec(state.adaptive_nodes.done_time)] / state.done_time[state.T]
  is_done = [state.is_done; zeros(num_actions)]
  is_transport = [repeat([1; 0], state.N_OPP); ones(state.N); 0; 0; repeat([1; 0], size(state.adaptive_nodes.done_time, 1))]
  is_opperation = [repeat([0; 1], state.N_OPP); zeros(state.N); 0; 0; repeat([0; 1], size(state.adaptive_nodes.done_time, 1))]
  is_source = [zeros(state.S-1); 1; zeros(length(state.adaptive_nodes.done_time))]
  is_sink = [zeros(state.T-1); 1; zeros(length(state.adaptive_nodes.done_time)+1)]
  return GNNGraph([state.conj_src; state.disj_src; vec(state.adaptive_nodes.src)],
    [state.conj_tar; state.disj_tar; vec(state.adaptive_nodes.tar)],
    num_nodes=state.S + num_actions,
    ndata=Float32.([done_time is_done is_transport is_opperation is_source is_sink]'))
end

#####
##### Interaction APIc
#####

function GI.action_string(spec::GameSpec, o)
  return string(o)
end

function GI.parse_action(spec::GameSpec, str)
  try
    s = split(str)
    return 1
  catch e
    return nothing
  end
end


function GI.read_state(::GameSpec)#input problem
  return nothing
end


function GI.render(g::GameEnv)
  m_opperation = zeros(UInt8, g.N_OPP)
  for (m, o) in enumerate(g.start_machine)
    print(crayon"white", "m", m, ":")
    last_done = 0
    while (!iszero(o))
      i = count(>(g.N_OPP * 2), g.conj_tar[1:o-1]) + 1
      idle = (crayon"dark_gray", repeat("-", g.done_time[o] - g.process_time[o÷2, m] - last_done))
      active = (Crayon(foreground=i), repeat("=", g.process_time[o÷2, m]))
      print(idle..., active...)
      last_done = g.done_time[o]
      m_opperation[o÷2] = m
      #get next operation for machine
      index = findfirst(isequal(o), g.disj_src)
      if (isnothing(index))
        break
      end
      o = g.disj_tar[index]
    end
    fill = (crayon"dark_gray", repeat("-", g.done_time[g.T] - last_done))
    print(fill...)
    println(crayon"reset")
  end
  for (k, t) in enumerate(g.start_vehicle)
    print(crayon"white", "k", k, ":")
    last_done = 0
    location = g.M + 1
    while (!iszero(t))
      if (t <= g.N_OPP * 2)
        i = count(>(g.N_OPP * 2), g.conj_tar[1:t]) + 1
        dropoff = m_opperation[(t+1)÷2]
      else
        i = Int64(g.T - t)
        dropoff = g.M + 1
      end
      pickup_index = findfirst(isequal(t), g.conj_tar)
      pickup = pickup_index > g.N_OPP * 2 ? g.M + 1 : m_opperation[pickup_index÷2]
      empty_time = g.transport_time[location, pickup]
      loaded_time = g.transport_time[pickup, dropoff]
      idle = (crayon"dark_gray", repeat("-", g.done_time[t] - last_done - empty_time - loaded_time))
      empty = (crayon"dark_gray", repeat("=", empty_time))
      loaded = (Crayon(foreground=i), repeat("=", loaded_time))
      print(idle..., empty..., loaded...)
      last_done = g.done_time[t]
      location = dropoff
      #get next transport for vehicle
      index = findfirst(isequal(t), g.disj_src)
      isnothing(index) && break
      t = g.disj_tar[index]
    end
    fill = (crayon"dark_gray", repeat("-", g.done_time[g.T] - last_done))
    print(fill...)
    println(crayon"reset")
  end
  for n in 1:g.N
    print(Crayon(foreground=n), "n", n, " ")
  end
  print(crayon"white", "makespan: ", g.done_time[g.T])
  print(crayon"white", " reward: ", GI.white_reward(g))
  println(crayon"reset")
  #print graph
  graph = GI.vectorize_state(GI.spec(g), g)
  membership = [repeat([1; 2], g.N_OPP); repeat([3], g.N); 4; 5; repeat([1; 2], size(g.adaptive_nodes.done_time, 1))]
  nodecolor = [colorant"deepskyblue1", colorant"royalblue4", colorant"orange", colorant"red", colorant"green"]
  nodefillc = nodecolor[membership]
  draw(PNG("games/fjspt/graphs/graph" * string(count(g.is_done[1:(g.N_OPP*2)]) ÷ 2) * ".png", 50cm, 50cm), gplot(graph, nodelabel=1:nv(graph), nodefillc=nodefillc))
end
