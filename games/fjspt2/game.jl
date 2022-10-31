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
  p_time = rand(rng, spec.P.first:spec.P.second, total_opps, M + 1)
  for o in 1:total_opps
    if (!(o + 1 in job_ids))#return operation
      not_able = [randperm(rng, M)[1:(M-rand(rng, spec.A.first[itc]:spec.A.second[itc]))]; M + 1]
      p_time[o, not_able] .= 0xff
    else
      p_time[o, 1:M] .= 0xff
      p_time[o, M+1] = 0
    end
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
    zeros(UInt16, M + 1),
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
  M = instance.num_machines
  N = length(instance.num_operations)
  K = instance.num_vehicles
  job_ids = cumsum([1; num_operations])
  num_nodes = vec(sum(p_time .!= 0xff, dims=2)) .+ K
  node_ids = cumsum([3; num_nodes])#S, T, nodes
  NUM_ACTIONS = job_ids[end] - 1
  return GameEnv(
    p_time,
    t_time,
    M,
    N,
    K,
    job_ids,
    sum(maximum(replace(p_time, 0xff => 0x00), dims=2)) + maximum(t_time) * (2 * NUM_ACTIONS), #All worst operations in a row,,
    sum(sum.(minimum.([p_time[job_ids[i]:job_ids[i+1]-1, :] for i in 1:length(job_ids)-1], dims=2))),
    #State
    zeros(UInt16, NUM_ACTIONS, 2),
    zeros(UInt16, NUM_ACTIONS, 2),
    zeros(UInt16, NUM_ACTIONS, 2),
    zeros(UInt16, NUM_ACTIONS, 2),
    #Info
    zeros(UInt16, M + 1),
    zeros(UInt16, K),
    node_ids
  )
end

GI.init(spec::GameSpec, s) = copy(s)

GI.spec(g::GameEnv) = GameSpec(ConstSchedule(g.M) => ConstSchedule(g.M), ConstSchedule(g.N) => ConstSchedule(g.N), ConstSchedule(2) => ConstSchedule(2), ConstSchedule(2) => ConstSchedule(2), 1 => 5, 1 => 5)

GI.two_players(::GameSpec) = false

GI.state_dim(spec::GameSpec) = (6, 0)

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

GI.game_terminated(g::GameEnv) = !any(g.assigned[g.job_ids[2:end].-1, 2] .== 0)

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
function next_nodes(g::GameEnv)
  nodes = Vector{UInt16}()
  for (j, o) in enumerate(g.job_ids[1:end-1])
    while (o != g.job_ids[j+1])
      if (g.assigned[o, 2] == 0)#machine unscheduled
        if (g.assigned[o, 1] == 0)#transport unscheduled
          append!(nodes, g.node_ids[o]:g.node_ids[o]+count(g.last_o_k .!= 0xff)-1)
        else
          append!(nodes, g.node_ids[o]+1:g.node_ids[o+1]-1)
        end
        break
      else
        o += 1
      end
    end
  end
  if(isempty(nodes))
    @show g.assigned
    @show g.previous
    @show g.K
    @show g.last_o_k
    @show count(g.last_o_k .!= 0xff)
  end
  return nodes
end

function GI.available_actions(g::GameEnv)
  return collect(1:length(next_nodes(g)))
end

function GI.actions_mask(g::GameEnv)
  return trues(length(next_nodes(g)))
end

function properties(action, node_ids)
  o = findfirst(>(action), node_ids) - 1
  local_index = action - node_ids[o] + 1
  return (o, local_index)
end

function GI.play!(g::GameEnv, action)
  action = next_nodes(g)[action]
  o, local_index = properties(action, g.node_ids)
  is_transport = (g.assigned[o, 1] == 0)
  is_first = o in g.job_ids
  prev_o_i = o - 1
  if (is_transport)
    k = findall(g.last_o_k .!= 0xff)[local_index]
    @assert k <= g.K
    prev_o_k = g.last_o_k[k]
    #get vehicle locations
    m_k = iszero(prev_o_k) ? g.M + 1 : g.assigned[prev_o_k, 2]
    m_i = is_first ? g.M + 1 : g.assigned[prev_o_i, 2]
    #update solution
    g.assigned[o, 1] = k
    #update time
    g.ready_time[o, 1] = max(is_first ? 0 : g.done_time[prev_o_i, 2], iszero(prev_o_k) ? 0 : g.done_time[prev_o_k, 1])
    #update ids
    g.node_ids[o+1:end] .-= count(g.last_o_k .!= 0xff) - 1
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
    @assert k <= g.K
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
    return ((g.UB - maximum(g.done_time[g.job_ids[2:end].-1, 2])) / (g.UB - g.LB))
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
  transport_nodes = g.assigned[o, 1] != 0 ? 1 : count(g.last_o_k .!= 0xff)
  transport = repeat([t], transport_nodes)
  machines = repeat([m], g.assigned[o, 2] != 0 ? 1 : g.node_ids[o+1] - g.node_ids[o] - transport_nodes)
  return [transport; machines]
end

function get_machines(g::GameEnv, o)
  is_first = o + 1 in g.job_ids
  m = is_first ? g.M + 1 : g.assigned[o, 2]
  m = m == 0 ? findall(!=(0xff), g.process_time[o, :]) : m
  return m
end

function time_for_action(g::GameEnv, o)
  t = g.assigned[o, 1]
  m = g.assigned[o, 2]
  if (t != 0 && m != 0)
    transport = Float32(g.done_time[o, 1] - g.ready_time[o, 1])
  elseif (t != 0)
    m_k = g.previous[o, 1] == 0 ? g.M + 1 : g.assigned[g.previous[o, 1], 2]
    prev_m = get_machines(g, o - 1)
    next_m = get_machines(g, o)
    transport = mean(g.transport_time[m_k, prev_m]) + mean(g.transport_time[prev_m, next_m])
  else
    transport = repeat([mean(g.transport_time) * 2], count(g.last_o_k .!= 0xff))
  end
  machines = m == 0 ? filter(!=(0xff), g.process_time[o, :]) : g.process_time[o, m]
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
      k = filter(!=(0xff), s.last_o_k)
      prev_conj = is_first ? [1] : (s.node_ids[o-1]+(s.assigned[o-1, 1] == 0 ? count(s.last_o_k .!= 0xff) : 1)):s.node_ids[o]-1
      next_conj = s.node_ids[o]:(s.node_ids[o]+length(k)-1)
      is_started = k .!= 0
      prev_disj = s.node_ids[k[is_started]]
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
  membership = reduce(vcat, [node_group(s, o, 2 * o - 1, 2 * o) for o in 1:s.job_ids[end]-1])
  is_done = [0.0; GI.game_terminated(s) ? 1.0 : 0.0; Float32.(vec(s.assigned' .!= 0))[membership]]
  is_transport = [0.0; 0.0; repeat([1.0, 0.0], s.job_ids[end] - 1)[membership]]
  ready_time = [0.0; maximum(s.ready_time); Float32.(vec(s.ready_time'))[membership]]
  done_time = [0.0; maximum(s.done_time); Float32.(vec(s.done_time'))[membership]]
  action_time = floor.([0.0; 0.0; [time_for_action(s, o) for o in 1:s.job_ids[end]-1]...])
  is_next_action = zeros(Float32, s.node_ids[end] - 1)
  is_next_action[next_nodes(s)] .= 1.0
  return GNNGraph(
    UInt16.([conj_src; disj_src]),
    UInt16.([conj_tar; disj_tar]),
    num_nodes=s.node_ids[end] - 1,
    ndata=UInt16.([is_done is_transport ready_time done_time action_time is_next_action]'))#
end

#####
##### Interaction APIc
#####

function GI.action_string(g::GameEnv, o)
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
  t = o == 0 ? 0 : g.done_time[o, datarow]
  while (o != 0)
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
  print(crayon"white", " lowerbound: ", g.LB * 1)
  print(crayon"white", " upperbound: ", g.UB * 1)
  print(crayon"white", " reward: ", GI.white_reward(g))
  println(crayon"reset")
  @show g.assigned * 1
  @show g.ready_time * 1
  @show g.done_time * 1
  @show g.last_o_k * 1
  @show g.last_o_m * 1
  #print graph
  graph = GI.vectorize_state(GI.spec(g), g)
  membership = [4; 5; [node_group(g, o, 2, 3) for o in 1:(g.job_ids[end]-1)]...]
  nodecolor = [colorant"deepskyblue1", colorant"lightblue", colorant"orange", colorant"red", colorant"green"]
  nodefillc = nodecolor[membership]
  draw(PNG("games/fjspt2/graphs/graph" * string(count(g.assigned .!= 0)) * ".png", 50cm, 50cm), gplot(graph, nodelabel=1:nv(graph), nodefillc=nodefillc))
end

function recalculate_bounds!(g::GameEnv)
  g.UB = sum(maximum(replace(g.process_time, 0xff => 0x00), dims=2)) + maximum(g.transport_time) * (2 * (g.job_ids[end] - 1)) #All worst operations in a row
  g.LB = sum(sum.(minimum.([g.process_time[g.job_ids[i]:g.job_ids[i+1]-1, :] for i in 1:length(g.job_ids)-1], dims=2)))
end

function add_job!(spec::GameSpec, g::GameEnv, rng::AbstractRNG, itc)
  new_operations = rand(rng, 2:(g.M+1))
  g.process_time = [g.process_time; rand(rng, spec.P.first:spec.P.second, new_operations, g.M + 1)]
  old_num_operations = g.job_ids[end] - 1
  for o in old_num_operations+1:old_num_operations+new_operations
    if (o != old_num_operations + new_operations)#return operations
      not_able = [randperm(rng, g.M)[1:(g.M-rand(rng, spec.A.first[itc]:spec.A.second[itc]))]; g.M + 1]
      g.process_time[o, not_able] .= 0xff
    else
      g.process_time[o, 1:g.M] .= 0xff
      g.process_time[o, g.M+1] = 0
    end
  end
  g.N += 1
  append!(g.job_ids, g.job_ids[end] + new_operations)
  g.assigned = [g.assigned; zeros(UInt16, new_operations, 2)]
  g.previous = [g.previous; zeros(UInt16, new_operations, 2)]
  g.ready_time = [g.ready_time; zeros(UInt16, new_operations, 2)]
  g.done_time = [g.done_time; zeros(UInt16, new_operations, 2)]
  num_nodes = vec(sum(g.process_time[end-new_operations+1:end, :] .!= 0xff, dims=2)) .+ count(g.last_o_k .!= 0xff)
  g.node_ids = [g.node_ids; cumsum([g.node_ids[end]; num_nodes])[2:end]]
  recalculate_bounds!(g)
  return true
end

function fix_last_o(o, first, last)
  if (o < first || o == 0xff)
    return o
  elseif (o <= last)
    return 0
  else
    return o - (last - first + 1)
  end
end

function fix_previous_o(o, first, last, previous)
  if (o < first)
    return o
  elseif (o <= last)
    return fix_previous_o(previous[o], first, last, previous)
  else
    return o - (last - first + 1)
  end
end
function remove_job!(spec::GameSpec, g::GameEnv, rng::AbstractRNG, itc)
  g.N == 1 && return false
  removed_job = rand(rng, 1:g.N)
  first_op = g.job_ids[removed_job]
  last_op = g.job_ids[removed_job+1] - 1
  removed_ops = last_op - first_op + 1
  kept_operations = [trues(first_op - 1); falses(removed_ops); trues(g.job_ids[end] - 1 - last_op)]
  g.process_time = g.process_time[kept_operations, :]
  g.N -= 1
  g.job_ids = [g.job_ids[1:removed_job-1]; (g.job_ids[removed_job+1:end] .- removed_ops)]
  g.assigned = g.assigned[kept_operations, :]
  g.previous[:, 1] = map(x -> fix_previous_o(x, first_op, last_op, g.previous[:, 1]), g.previous[:, 1])
  g.previous[:, 2] = map(x -> fix_previous_o(x, first_op, last_op, g.previous[:, 2]), g.previous[:, 2])
  g.previous = g.previous[kept_operations, :]
  g.ready_time = g.ready_time[kept_operations, :]
  g.done_time = g.done_time[kept_operations, :]
  g.last_o_k = map(x -> fix_last_o(x, first_op, last_op), g.last_o_k)
  g.last_o_m = map(x -> fix_last_o(x, first_op, last_op), g.last_o_m)
  removed_nodes = g.node_ids[last_op+1] - g.node_ids[first_op]
  g.node_ids = [g.node_ids[1:first_op-1]; (g.node_ids[last_op+1:end] .- removed_nodes)]
  recalculate_bounds!(g)
  return true
end

function add_machine!(spec::GameSpec, g::GameEnv, rng::AbstractRNG, itc)
  num_ops = g.job_ids[end] - 1
  not_able = rand(rng, Bool, num_ops)#50/50 chance
  g.process_time = [g.process_time[:, 1:g.M] rand(rng, spec.P.first:spec.P.second, num_ops) g.process_time[:, end]]
  g.process_time[not_able, g.M+1] .= 0xff
  g.transport_time = [g.transport_time[1:g.M, 1:g.M] rand(rng, spec.T.first:spec.T.second, g.M) g.transport_time[1:g.M, g.M+1]
    rand(rng, spec.T.first:spec.T.second, 1, g.M) 0 rand(rng, spec.T.first:spec.T.second)
    g.transport_time[g.M+1, 1:g.M]' rand(rng, spec.T.first:spec.T.second) 0]
  g.M += 1
  g.last_o_m = [g.last_o_m[1:g.M]; 0; g.last_o_m[end]]
  is_affected = (g.assigned[:, 2] .== 0) .&& (g.process_time[:, g.M] .!= 0xff)
  g.node_ids += cumsum([false; is_affected])
  recalculate_bounds!(g)
  return true
end

function remove_machine!(spec::GameSpec, g::GameEnv, rng::AbstractRNG, itc)
  removed_machine = rand(rng, 1:g.M)
  keep_machines = trues(g.M + 1)
  keep_machines[removed_machine] = false
  if (any(sum(g.process_time[g.assigned[:, 2].==0, removed_machine] .!= 0xff, dims=2) .== 0))
    return false #if no other machine available for future operation, abort
  end
  g.process_time[:, removed_machine] .= 0xff
  is_affected = (g.assigned[:, 2] .== 0) .&& (g.process_time[:, removed_machine] .!= 0xff)
  g.node_ids -= cumsum([false; is_affected])

  # g.transport_time = g.transport_time[keep_machines, keep_machines]
  # g.M -= 1
  # g.last_o_m = g.last_o_m[keep_machines]
  # is_affected = (g.assigned[:, 2] .== 0) .&& (g.process_time[:, removed_machine] .!= 0xff)
  # g.node_ids -= cumsum([false; is_affected])
  # g.process_time = g.process_time[:, keep_machines]
  recalculate_bounds!(g)
  return true
end

function add_vehicle!(spec::GameSpec, g::GameEnv, rng::AbstractRNG, itc)
  g.K += 1
  append!(g.last_o_k, 0)
  is_affected = g.assigned[:, 1] .== 0
  g.node_ids += cumsum([false; is_affected])
  return true
end

function remove_vehicle!(spec::GameSpec, g::GameEnv, rng::AbstractRNG, itc)
  count(g.last_o_k .!= 0xff) == 1 && return false
  removed_vehicle = rand(rng, findall(g.last_o_k .!= 0xff))
  g.last_o_k[removed_vehicle] = 0xff
  # g.K -= 1
  # deleteat!(g.last_o_k, removed_vehicle)
  #Revert non-executed transport
  transport_planned = (g.assigned[:, 1] .== removed_vehicle) .&& (g.assigned[:, 2] .== 0)
  g.assigned[transport_planned, 1] .= 0
  g.ready_time[transport_planned, 1] .= 0
  g.node_ids += cumsum([false; transport_planned]) * count(g.last_o_k .!= 0xff)
  #Fix node ids
  is_affected = g.assigned[:, 1] .== 0
  g.node_ids -= cumsum([false; is_affected])
  return true
end

function change_process_time!(spec::GameSpec, g::GameEnv, rng::AbstractRNG, itc)
  o = rand(rng, 1:(g.job_ids[end]-1))
  m = rand(rng, findall(!=(0xff), g.process_time[o, :]))
  g.process_time[o, m] = rand(rng, spec.P.first:spec.P.second)
  recalculate_bounds!(g)
  return true
end

function change_transport_time!(spec::GameSpec, g::GameEnv, rng::AbstractRNG, itc)
  m1 = rand(rng, 1:g.M)
  m2 = rand(rng, 1:g.M)
  if (m1 != m2)
    g.transport_time[m1, m2] = rand(rng, spec.T.first:spec.T.second)
    recalculate_bounds!(g)
    return true
  end
  return false
end

function GI.disturbe!(spec::GameSpec, g::GameEnv, rng::AbstractRNG, itc)
  if (rand(rng) < 0 / 10)
    type = rand(rng, 1:8)
    @show type
    if (type == 1)#Job arrival
      result = add_job!(spec, g, rng, itc)
    elseif (type == 2)#Job cancelation
      result = remove_job!(spec, g, rng, itc)
    elseif (type == 3)#Machine repair 
      result = add_machine!(spec, g, rng, itc)
    elseif (type == 4)#Machine breakdown
      result = remove_machine!(spec, g, rng, itc)
    elseif (type == 5)#Vehicle repair 
      result = add_vehicle!(spec, g, rng, itc)
    elseif (type == 6)#Vehicle breakdown
      result = remove_vehicle!(spec, g, rng, itc)
    elseif (type == 7)#Process time change
      result = change_process_time!(spec, g, rng, itc)
    elseif (type == 8)#Transport time change  
      result = change_transport_time!(spec, g, rng, itc)
    else
      result = false
    end
    return result
  end
  return false
end