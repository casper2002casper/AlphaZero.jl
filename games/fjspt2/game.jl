import AlphaZero.GI
import Cairo, Fontconfig
using AlphaZero: AbstractSchedule, ConstSchedule
using GraphNeuralNetworks
using Random
using Crayons
using JSON3
using StructTypes
using GraphPlot, Compose, Graphs, Colors
using Statistics: mean

struct GameSpec <: GI.AbstractGameSpec
  M::Pair{AbstractSchedule,AbstractSchedule}
  N::Pair{AbstractSchedule,AbstractSchedule}
  A::Pair{AbstractSchedule,AbstractSchedule}
  K::Pair{AbstractSchedule,AbstractSchedule}
  P::Pair{UInt8,UInt8}
  T::Pair{UInt8,UInt8}
end

mutable struct GameEnv <: GI.AbstractGameEnv
  #Problem instance
  process_time::Matrix{UInt8} #o,m -> p_time
  transport_time::Matrix{UInt8} #m, m' -> t_time
  M::UInt8
  N::UInt8
  K::UInt8
  job_ids::Vector{UInt8} # i -> operations
  UB::UInt16
  LB::UInt16
  #Solution
  assigned::Matrix{UInt16} #o -> m,k
  previous::Matrix{UInt16} #o -> o_m, o_k
  ready_time::Matrix{UInt16} #o -> setup_done
  done_time::Matrix{UInt16} #o -> o_done
  #Info 
  last_o_m::Vector{UInt16}#m->o
  last_o_k::Vector{UInt16}#k->o
  node_ids::Vector{UInt16}#o->e
end

function Base.hash(s::GameEnv, h::UInt)
  return hash([s.assigned; s.previous], h)
end

Base.isequal(a::GameEnv, b::GameEnv) = isequal((a.assigned, a.previous), (b.assigned, b.previous))

Base.copy(s::GameEnv) = GameEnv(
  #Static values
  s.process_time,
  s.transport_time,
  s.M,
  s.N,
  s.K,
  s.job_ids,
  s.UB,
  s.LB,
  #Mutable values
  copy(s.assigned),
  copy(s.previous),
  copy(s.ready_time),
  copy(s.done_time),
  copy(s.last_o_m),
  copy(s.last_o_k),
  copy(s.node_ids)
)

function GI.init(spec::GameSpec, itc::Int, rng::AbstractRNG)
  N = rand(rng, UInt8(spec.N.first[itc]):UInt8(spec.N.second[itc]))
  M = rand(rng, UInt8(spec.M.first[itc]):UInt8(spec.M.second[itc]))
  K = rand(rng, UInt8(spec.K.first[itc]):UInt8(spec.K.second[itc]))
  num_operations = rand(rng, 2:(M+1), N)
  job_ids = cumsum([1; num_operations])
  total_opps = sum(num_operations)
  p_time = rand(rng, spec.P.first:spec.P.second, total_opps, M+1);
  for o in 1:total_opps
    is_return_opperation = o+1 in job_ids
    if(!is_return_opperation)
      not_able = randperm(rng, M)[1:M-rand(rng, spec.A.first[itc]:spec.A.second[itc])]
      p_time[o, not_able] .= 0xff
    else
      p_time[o, :] .= 0xff
    end 
    p_time[o, M+1] = is_return_opperation ? 0 : 0xff
  end
  t_time = rand(rng, spec.T.first:spec.T.second, M + 1, M + 1)
  for m in 1:M+1
    t_time[m, m] = 0x00
  end
  num_nodes = vec(sum(p_time .!= 0xff, dims=2)) .+ K
  node_ids = cumsum([3; num_nodes])#S, T, nodes
  NUM_ACTIONS = job_ids[end] - 1 
  return GameEnv(
    #Problem instance
    p_time,
    t_time,
    M,
    N,
    K,
    job_ids,
    sum(maximum(replace(p_time, 0xff => 0x00), dims=2)) + maximum(t_time) * (2 * NUM_ACTIONS), #All worst operations in a row,,
    sum(sum.(minimum.([p_time[job_ids[i]:job_ids[i+1]-1, :] for i in 1:length(job_ids)-1], dims=2))),
    #Solution
    zeros(UInt16, NUM_ACTIONS, 2),
    zeros(UInt16, NUM_ACTIONS, 2),
    zeros(UInt16, NUM_ACTIONS, 2),
    zeros(UInt16, NUM_ACTIONS, 2),
    #Info
    zeros(UInt16, M+1),
    zeros(UInt16, K),
    node_ids
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

function GI.init(spec::GameSpec, instance_string::String) #fix
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
  num_nodes = sum(p_time .!= 0xff, dims=1) .+ K
  node_ids = cumsum(num_nodes)
  return GameEnv(
    p_time,
    t_time,
    instance.num_machines,
    N,
    instance.num_vehicles,
    instance.num_operations,
    sum(maximum(replace(p_time, 0xff => 0x00), dims=2)) + maximum(t_time) * (2 * N_OPP + N), #All worst operations in a row,
    node_done[T],
    #State
    zeros(UInt16, NUM_OPP * 2 + N, 2),
    zeros(UInt16, NUM_OPP * 2 + N, 2),
    zeros(UInt16, NUM_OPP * 2 + N),
    zeros(UInt16, NUM_OPP * 2 + N),
    #Info
    zeros(UInt16, M),
    zeros(UInt16, K),
    node_ids
  )
end

GI.init(spec::GameSpec, s) = copy(s)

GI.spec(g::GameEnv) = GameSpec(ConstSchedule(g.M) => ConstSchedule(g.M), ConstSchedule(g.N) => ConstSchedule(g.N), ConstSchedule(2) => ConstSchedule(2), ConstSchedule(2) => ConstSchedule(2), 1 => 5, 1 => 5)

GI.two_players(::GameSpec) = false

GI.state_dim(spec::GameSpec) = (5, 0)

GI.state_type(spec::GameSpec) = return GameEnv

function GI.set_state!(g::GameEnv, s)
  g.assigned = copy(s.assigned)
  g.previous = copy(s.previous)
  g.ready_time = copy(s.ready_time)
  g.done_time = copy(s.done_time)
  g.last_o_m = copy(s.last_o_m)
  g.last_o_k = copy(s.last_o_k)
  g.node_ids = copy(s.node_ids)
end

GI.current_state(g::GameEnv) = copy(g)

GI.white_playing(g::GameEnv) = true

GI.game_terminated(g::GameEnv) = !any(g.assigned[g.job_ids[2:end].-1,2] .== 0)

GI.actions(spec::GameSpec) = []

# function next_action(g::GameEnv, i)
#   o = g.last_o_i[i] + 1
#   if (o == g.job_ids[i+1])
#     return []
#   end
#   if (g.assigned[o, 1] == 0)#transport unscheduled
#     return g.node_ids[o]:g.node_ids[o]+g.K-1
#   else
#     return g.node_ids[o]+1:g.node_ids[o+1]-1
#   end
# end

# function GI.available_actions(g::GameEnv)
#   return reduce(vcat, [next_action(g, i) for i in 1:g.N])
# end

function GI.available_actions(g::GameEnv)
  nodes = Vector{UInt16}()
  for (j, o) in enumerate(g.job_ids[1:end-1])
    while (o != g.job_ids[j+1])
      if (g.assigned[o, 2] == 0)#machine unscheduled
        if (g.assigned[o, 1] == 0)#transport unscheduled
          append!(nodes, g.node_ids[o]:g.node_ids[o]+g.K-1)
        else
          append!(nodes, g.node_ids[o]+1:g.node_ids[o+1]-1)
        end
        break
      else
        o += 1
      end
    end
  end
  return nodes
end

function GI.actions_mask(g::GameEnv)
  mask = falses(g.node_ids[end] - 1)
  available = GI.available_actions(g)
  mask[available] .= true
  return mask
end

function properties(action, node_ids)
  o = findfirst(>(action), node_ids) - 1
  local_index = action - node_ids[o] + 1
  return (o, local_index)
end

function GI.play!(g::GameEnv, action)
  o, local_index = properties(action, g.node_ids)
  is_transport = (g.assigned[o, 1] == 0)
  is_first = o in g.job_ids
  prev_o_i = o - 1
  if (is_transport)
    k = local_index
    prev_o_k = g.last_o_k[k]
    #get vehicle locations
    m_k = iszero(prev_o_k) ? g.M + 1 : g.assigned[prev_o_k, 2]
    m_i = is_first ? g.M + 1 : g.assigned[prev_o_i, 2]
    #update solution
    g.assigned[o, 1] = k
    #update time
    g.ready_time[o, 1] = max(is_first ? 0 : g.done_time[prev_o_i, 2], iszero(prev_o_k) ? 0 : g.done_time[prev_o_k, 1])
    #update ids
    g.node_ids[o+1:end] .-= g.K - 1
  else
    m_o = findall(!=(0xff), g.process_time[o, :])
    m = m_o[local_index-1]
    prev_o_m = g.last_o_m[m]
    #update solution
    g.assigned[o, 2] = m
    g.previous[o, 2] = prev_o_m
    #update info vectors
    g.last_o_m[m] = o
    #update vehicle
    k = g.assigned[o, 1]
    prev_o_k = g.last_o_k[k]
    m_k = iszero(prev_o_k) ? g.M + 1 : g.assigned[prev_o_k, 2]
    m_i = is_first ? g.M + 1 : g.assigned[prev_o_i, 2]
    transport_needed = m_i != m
    transport_needed && (g.last_o_k[k] = o)
    on_location = max(g.ready_time[o, 1], iszero(prev_o_k) ? 0 : g.done_time[prev_o_k, 1] + (transport_needed ? g.transport_time[m_k, m_i] : 0))
    g.previous[o, 1] = prev_o_k
    g.done_time[o, 1] = max(iszero(prev_o_m) ? 0 : g.done_time[prev_o_m, 2], on_location + g.transport_time[m_i, m])
    #update time
    g.ready_time[o, 2] = max(g.done_time[o, 1], iszero(prev_o_m) ? 0 : g.done_time[prev_o_m, 2])
    g.done_time[o, 2] = g.ready_time[o, 2] + g.process_time[o, m]
    #update ids
    g.node_ids[o+1:end] .-= length(m_o) - 1
  end
end

function GI.white_reward(g::GameEnv)
  if (GI.game_terminated(g))
    return ((g.UB - maximum(g.done_time[g.job_ids[2:end].-1,2])) / (g.UB - g.LB))
  end
  return 0
end

function add_conj_nodes!(previous_nodes, next_nodes, conj_src, conj_tar)
  for previous in previous_nodes
    append!(conj_src, next_nodes)
    append!(conj_tar, repeat([previous], length(next_nodes)))
  end
end

function node_group(g::GameEnv, o, t, m)
  transport_nodes = g.assigned[o, 1] != 0 ? 1 : g.K
  transport = repeat([t], transport_nodes)
  machines = repeat([m], g.assigned[o, 2] != 0 ? 1 : g.node_ids[o+1] - g.node_ids[o] - transport_nodes)
  return [transport; machines]
end

function get_machines(g::GameEnv, o)
  is_first = o+1 in g.job_ids
  m = is_first ? g.M+1 : g.assigned[o, 2]
  m = m == 0 ? findall(!=(0xff), g.process_time[o, :]) : m
  return m
end

function time_for_action(g::GameEnv, o)
  t = g.assigned[o, 1] 
  m = g.assigned[o, 2]
  if(t != 0 && m != 0)
    transport = Float32(g.done_time[o,1] - g.ready_time[o,1])
  elseif (t != 0)
    m_k = g.previous[o, 1] == 0 ? g.M+1 : g.assigned[g.previous[o, 1], 2]
    prev_m = get_machines(g, o-1)
    next_m = get_machines(g, o)
    transport = mean(g.transport_time[m_k, prev_m]) + mean(g.transport_time[prev_m, next_m])
  else
    transport = repeat([mean(g.transport_time)*2], 2)
  end
  machines = m == 0 ? filter(!=(0xff) ,g.process_time[o,:]) : g.process_time[o,m]
  return [transport; machines]
end

function GI.vectorize_state(::GameSpec, s::GameEnv)
  conj_src = Vector{UInt16}()
  conj_tar = Vector{UInt16}()
  disj_src = Vector{UInt16}()
  disj_tar = Vector{UInt16}()
  for o in 1:(s.job_ids[end]-1)
    is_first = o in s.job_ids
    is_last = o + 1 in s.job_ids
    #Transport
    if (s.assigned[o, 1] != 0)#scheduled
      prev_conj = [is_first ? 1 : s.node_ids[o] - 1]
      next_conj = [s.node_ids[o]]
      prev_disj = iszero(s.previous[o, 1]) ? [] : [s.node_ids[s.previous[o, 1]]]
      next_disj = iszero(s.previous[o, 1]) ? [] : next_conj
    else#unscheduled
      prev_conj = is_first ? [1] : (s.node_ids[o-1]+(s.assigned[o-1, 1] == 0 ? s.K : 1)):s.node_ids[o]-1
      next_conj = s.node_ids[o]:(s.node_ids[o]+s.K-1)
      is_started = s.last_o_k .!= 0
      prev_disj = s.node_ids[s.last_o_k[is_started]] 
      next_disj = next_conj[is_started]
    end
    add_conj_nodes!(prev_conj, next_conj, conj_src, conj_tar)
    append!(disj_src, prev_disj)
    append!(disj_tar, next_disj)
    #Machines
    if (s.assigned[o, 2] != 0)#scheduled
      prev_conj = [s.node_ids[o]]
      next_conj = [s.node_ids[o] + 1]
      prev_disj = iszero(s.previous[o, 2]) ? [] : s.node_ids[s.previous[o, 2]] + 1
      next_disj = iszero(s.previous[o, 2]) ? [] : next_conj
    else
      m = findall(!=(0xff), s.process_time[o, :])
      prev_conj = s.node_ids[o]:(s.node_ids[o+1]-length(m)-1)
      next_conj = (s.node_ids[o+1]-length(m)):(s.node_ids[o+1]-1)
      is_started = s.last_o_m[m] .!= 0
      prev_disj = s.node_ids[s.last_o_m[m][is_started]] .+ 1 
      next_disj = next_conj[is_started]
    end
    add_conj_nodes!(prev_conj, next_conj, conj_src, conj_tar)
    append!(disj_src, prev_disj)
    append!(disj_tar, next_disj)
    is_last && add_conj_nodes!(next_conj, [2], conj_src, conj_tar)
  end
  membership = reduce(vcat,[node_group(s, o, 2*o-1, 2*o) for o in 1:s.job_ids[end]-1])
  is_done = [0.0;GI.game_terminated(s) ? 1.0 : 0.0;Float32.(vec(s.assigned' .!= 0))[membership]]
  is_transport = [0.0;0.0;repeat([1.0,0.0], s.job_ids[end]-1)[membership]]
  ready_time = [0.0;maximum(s.ready_time);Float32.(vec(s.ready_time'))[membership]]
  done_time = [0.0;maximum(s.done_time);Float32.(vec(s.done_time'))[membership]]
  action_time = [0.0;0.0; [time_for_action(s, o) for o in 1:s.job_ids[end]-1]...]
  return GNNGraph(
    [conj_src; disj_src],
    [conj_tar; disj_tar],
    num_nodes=s.node_ids[end] - 1,
    ndata=[is_done is_transport ready_time done_time action_time]')#
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

function print_schedule(g::GameEnv, id, is_machine)
  print(crayon"white", is_machine ? "m" : "k", id, ":")
  line = []
  datarow = is_machine ? 2 : 1
  o = is_machine ? g.last_o_m[id] : g.last_o_k[id]
  t = o==0 ? 0 : g.done_time[o, datarow]
  while(o != 0)
    i = count(<=(o), g.job_ids)
    append!(line, [repeat("-", t - g.done_time[o, datarow]); crayon"dark_gray"])
    append!(line, [repeat("=", g.done_time[o, datarow] - g.ready_time[o, datarow]); Crayon(foreground=i)])
    t = g.ready_time[o, datarow]
    o = g.previous[o, datarow]
  end
  append!(line, [repeat("-", t); crayon"dark_gray"])
  print(reverse(line)...)
  println(crayon"reset")
end

function GI.render(g::GameEnv)
  for m in 1:g.M
    print_schedule(g, m, true)
  end
  for k in 1:g.K
    print_schedule(g, k, false)
  end
  for n in 1:g.N
    print(Crayon(foreground=n), "n", n, " ")
  end
  print(crayon"white", "makespan: ", maximum(g.done_time))
  print(crayon"white", " lowerbound: ", g.LB*1)
  print(crayon"white", " upperbound: ", g.UB*1)
  print(crayon"white", " reward: ", GI.white_reward(g))
  println(crayon"reset")
  @show g.assigned * 1
  @show g.ready_time * 1
  @show g.done_time * 1
  @show g.last_o_k*1
  @show g.last_o_m*1
  #print graph
  graph = GI.vectorize_state(GI.spec(g), g)
  membership = [4; 5; [node_group(g, o, 2, 3) for o in 1:(g.job_ids[end]-1)]...]
  nodecolor = [colorant"deepskyblue1", colorant"lightblue", colorant"orange", colorant"red", colorant"green"]
  nodefillc = nodecolor[membership]
  draw(PNG("games/fjspt2/graphs/graph" * string(count(g.assigned .!= 0)) * ".png", 50cm, 50cm), gplot(graph, nodelabel=1:nv(graph), nodefillc=nodefillc))
end
