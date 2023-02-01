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
  #Solution
  assigned::Matrix{UInt16} #o -> m,k
  ready_time::Matrix{UInt16} #o -> setup_done
  done_time::Matrix{UInt16} #o -> o_done
  #Info 
  last_o_m::Vector{UInt16}#m->o
  last_o_k::Vector{UInt16}#k->o
end

function Base.hash(s::GameEnv, h::UInt)
  return hash([s.assigned; s.done_time], h)
end

Base.isequal(a::GameEnv, b::GameEnv) = isequal((a.assigned, a.done_time), (b.assigned, b.done_time))

Base.copy(s::GameEnv) = GameEnv(
  #Static values
  s.process_time,
  s.transport_time,
  s.M,
  s.N,
  s.K,
  s.job_ids,
  #Mutable values
  copy(s.assigned),
  copy(s.ready_time),
  copy(s.done_time),
  copy(s.last_o_m),
  copy(s.last_o_k),
)

function GI.init(spec::GameSpec, itc::Int, rng::AbstractRNG)
  N = rand(rng, UInt8(spec.N.first[itc]):UInt8(spec.N.second[itc]))
  M = rand(rng, UInt8(spec.M.first[itc]):UInt8(spec.M.second[itc]))
  K = rand(rng, UInt8(spec.K.first[itc]):UInt8(spec.K.second[itc]))
  num_operations = rand(rng, 2:(M+1), N)
  job_ids = cumsum([1; num_operations])
  total_opps = sum(num_operations)
  p_time = [
    rand(rng, spec.P.first:spec.P.second, total_opps, M) fill(0xFF, total_opps)
    fill(0xFF, 2, M) zeros(2, 1)]
  for o in 1:total_opps
    is_return_opperation = o + 1 in job_ids
    if (!is_return_opperation)
      not_able = randperm(rng, M)[1:M-rand(rng, spec.A.first[itc]:spec.A.second[itc])]
      p_time[o, not_able] .= 0xFF
    else
      p_time[o, :] = [fill(0xFF, M); 0]
    end
  end
  t_time = rand(rng, spec.T.first:spec.T.second, M + 1, M + 1)
  for m in 1:M+1
    t_time[m, m] = 0x00
  end
  assigned = zeros(UInt16, total_opps + 2, 2)
  assigned[total_opps+1, 2] = M + 1
  return GameEnv(
    #Problem instance
    p_time,
    t_time,
    M,
    N,
    K,
    job_ids,
    #Solution
    assigned,
    zeros(UInt16, total_opps + 2, 2),
    zeros(UInt16, total_opps + 2, 2),
    #Info
    fill(total_opps + 1, M + 1),
    fill(total_opps + 1, K)
  )
end

function GI.setup!(g::GameEnv)
  foreach(o -> propage_done_time(g, o, g.M + 1), g.job_ids[1:end-1])
  g.ready_time[g.job_ids[end]+1, :] = g.done_time[g.job_ids[end]+1, :] .= maximum(g.done_time[g.job_ids[2:end].-1, 2])
end

struct FJSPTInstance
  num_vehicles::Int
  num_machines::Int
  num_operations::Vector{Int}
  process_time::Vector{Int}
  transport_time::Vector{Int}
end

StructTypes.StructType(::Type{FJSPTInstance}) = StructTypes.Struct()

function propage_done_time(g::GameEnv, o, m_old)
  while (true)
    m = findall(!=(0xFF), g.process_time[o, :])
    g.ready_time[o, 1] = m_old == g.M + 1 ? 0 : g.done_time[o-1, 2]
    g.ready_time[o, 2] = g.done_time[o, 1] = g.ready_time[o, 1] + floor(mean(g.transport_time[m_old, m]))
    g.done_time[o, 2] = g.ready_time[o, 2] + floor(mean(g.process_time[o, m]))
    m == [g.M + 1] && break
    m_old = m
    o += 1
  end
end

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
    #sum(maximum(replace(p_time, 0xff => 0x00), dims=2)) + maximum(t_time) * (2 * N_OPP + N), #All worst operations in a row,
    #node_done[T],
    #State
    zeros(UInt16, NUM_OPP * 2 + N, 2),
    zeros(UInt16, NUM_OPP * 2 + N),
    zeros(UInt16, NUM_OPP * 2 + N),
    #Info
    zeros(UInt16, M),
    zeros(UInt16, K),
  )
end

GI.init(spec::GameSpec, s) = copy(s)

GI.spec(g::GameEnv) = GameSpec(ConstSchedule(g.M) => ConstSchedule(g.M), ConstSchedule(g.N) => ConstSchedule(g.N), ConstSchedule(2) => ConstSchedule(2), ConstSchedule(2) => ConstSchedule(2), 1 => 5, 1 => 5)

GI.two_players(::GameSpec) = false

GI.state_dim(spec::GameSpec) = (9, 1)

GI.state_type(spec::GameSpec) = return GameEnv

function GI.set_state!(g::GameEnv, s)
  g.assigned = copy(s.assigned)
  g.ready_time = copy(s.ready_time)
  g.done_time = copy(s.done_time)
  g.last_o_m = copy(s.last_o_m)
  g.last_o_k = copy(s.last_o_k)
end

GI.current_state(g::GameEnv) = copy(g)

GI.white_playing(g::GameEnv) = true

GI.game_terminated(g::GameEnv) = !any(g.assigned[g.job_ids[2:end].-1, 2] .== 0)

GI.actions(spec::GameSpec) = []

function next_operations(g::GameEnv)
  isnt_scheduled = g.assigned[:, 2] .== 0
  operations = Vector{UInt16}()
  for i in eachindex(g.job_ids[1:end-1])
    index = findfirst(isnt_scheduled[g.job_ids[i]:g.job_ids[i+1]-1])
    isnothing(index) || append!(operations, index - 1 + g.job_ids[i])
  end
  return operations
end

function GI.available_actions(g::GameEnv)
  next = next_operations(g)
  machines = g.process_time[next, :] .!= 0xFF
  num_machines = sum(machines)
  num_actions = num_machines * count(g.last_o_k .!= g.job_ids[end] + 1)
  return (1:num_actions)
end

function GI.actions_mask(g::GameEnv)
  actions = GI.available_actions(g)
  return trues(actions.stop)
end

function properties(g::GameEnv, action)
  num_vehicles = count(g.last_o_k .!= g.job_ids[end] + 1)
  for o in next_operations(g)
    machines_available = g.process_time[o, :] .!= 0xFF
    num_machines = sum(machines_available)
    if (action <= num_machines * num_vehicles)
      machines = findall(machines_available)
      m_i, k = divrem(action - 1, num_vehicles) .+ 1
      return o, machines[m_i], k
    end
    action -= num_machines * num_vehicles
  end
end

function GI.play!(g::GameEnv, action)
  o, m, k = properties(g, action)
  is_first = o in g.job_ids
  is_last = o+1 in g.job_ids
  prev_o_i = is_first ? g.job_ids[end] : o - 1
  prev_o_m = g.last_o_m[m]
  prev_o_k = g.last_o_k[k]
  #get vehicle locations
  m_k = g.assigned[prev_o_k, 2]
  m_i = g.assigned[prev_o_i, 2]
  #update info vectors
  g.last_o_m[m] = o
  transport_needed = m_i != m
  transport_needed && (g.last_o_k[k] = o)
  #update solution
  g.assigned[o, :] = [transport_needed ? k : 0; m]
  #update time
  g.ready_time[o, 1] = max(g.done_time[prev_o_i, 2], (transport_needed ? g.done_time[prev_o_k, 1] + g.transport_time[m_k, m_i] : 0))
  g.done_time[o, 1] = g.ready_time[o, 1] + g.transport_time[m_i, m]
  g.ready_time[o, 2] = max(g.done_time[o, 1], g.done_time[prev_o_m, 2])
  g.done_time[o, 2] = g.ready_time[o, 2] + g.process_time[o, m]
  #propagate done time
  is_last || propage_done_time(g, o + 1, m)
  g.ready_time[g.job_ids[end]+1, :] = g.done_time[g.job_ids[end]+1, :] .= maximum(g.done_time[g.job_ids[2:end].-1, 2])
end

function GI.white_reward(g::GameEnv)
  if (GI.game_terminated(g))
    #return ((g.UB - g.done_time[g.job_ids[end]+1, 2]) / (g.UB - g.LB))
    return -Float32(g.done_time[g.job_ids[end]+1, 2])
  end
  return 0
end

function GI.vectorize_state(::GameSpec, s::GameEnv)
  num_operations = size(s.process_time)[1]
  M = s.M + 1 #include virtual machine
  num_nodes = num_operations + M + s.K
  #data
  node_data = zeros(Float32, 9, num_nodes)
  node_data[1, :] = [zeros(Float32, num_operations); zeros(Float32, M + s.K)] #is operation
  node_data[2, :] = [zeros(Float32, num_operations); ones(Float32, M); zeros(s.K)] #is machine
  node_data[3, :] = [zeros(Float32, num_operations + M); ones(Float32, s.K)] #is vehicle 
  next_op = next_operations(s)
  next_op_mask = falses(num_operations)
  next_op_mask[next_op] .= true
  node_data[4, :] = [next_op_mask; zeros(Float32, M + s.K)] #next operation
  machine_available = s.process_time .!= 0xFF
  is_scheduled = s.assigned[:, 2] .!= 0
  machine_available[is_scheduled, :] .= false
  opp_left_m = sum(machine_available, dims=1)'
  opp_left_m[M] -= 1 #dont count s/t
  all_opp_done_m = vec(opp_left_m .== 0)
  node_data[5, :] = [is_scheduled; all_opp_done_m; zeros(s.K)] #is fully scheduled
  node_data[6, :] = [s.ready_time[:, 1]; s.done_time[s.last_o_m, 1]; s.done_time[s.last_o_k, 2]]#ready time
  opp_left_i = [sum(s.assigned[s.job_ids[i]:s.job_ids[i+1]-1, 2] .== 0) for i in 1:s.N]
  job_per_op = vcat([fill(i, s.job_ids[i+1] - s.job_ids[i]) for i in 1:s.N]...)
  node_data[7, :] = [opp_left_i[job_per_op]; 0.0; sum(opp_left_i); opp_left_m; fill(sum(opp_left_m), s.K)] #operations left
  process_left = copy(s.process_time)
  process_left[.!machine_available] .= 0
  possible_m = .!all_opp_done_m
  node_data[8, :] = [s.done_time[:, 2] .- s.ready_time[:, 1]; mean(process_left, dims=1)'; fill(mean(s.transport_time[s.assigned[s.last_o_k, 2], possible_m]) + mean(s.transport_time[possible_m, possible_m]), s.K)]#avg duration
  machine_per_o = sum(machine_available, dims=2)
  machine_per_o[machine_per_o.==0] .= 1
  node_data[9, :] = [machine_per_o; opp_left_m; fill(sum(opp_left_m), s.K)] #number of neighbours
  #edges
  src = Vector{UInt16}()
  tar = Vector{UInt16}()
  edge_data = Vector{Float32}()
  for o in 1:(s.job_ids[end]-1)
    is_first_op = o in s.job_ids
    is_last_op = o in (s.job_ids .- 1)
    #conj edges
    if (is_first_op)
      append!(src, num_operations - 1)
      append!(tar, o)
      append!(edge_data, 0)
    elseif (is_last_op)
      append!(src, [o - 1; o])
      append!(tar, [o; num_operations])
      append!(edge_data, [0; 0])
    else
      append!(src, o - 1)
      append!(tar, o)
      append!(edge_data, 0)
    end
    #machines
    if (s.assigned[o, 2] != 0)
      k, m = s.assigned[o, :]
      if(k == 0)
        k = []
        prev_m_k = []
      else
        valid_o_k = findall(i -> (s.assigned[i, 1] == k && s.done_time[i, 1] <= s.ready_time[o, 1]), 1:(s.job_ids[end]-1)) 
        prev_m_k = isempty(valid_o_k) ? M : s.assigned[argmax(o -> s.done_time[o, 1], valid_o_k), 2] 
      end
      prev_m = is_first_op ? M : s.assigned[o-1, 2]
    else
      m = findall(!=(0xFF), s.process_time[o, :])
      k = 1:s.K
      prev_m_k = s.assigned[s.last_o_k[k], 2]
      prev_m = is_first_op ? M : findall(!=(0xFF), s.process_time[o-1, :])
    end
    o_mk_src = fill(o, length(m) + length(k))
    o_mk_tar = [m; k .+ M] .+ num_operations
    time_to_m = s.transport_time[prev_m_k, prev_m]
    o_mk_data = [s.process_time[o, m]; length(time_to_m) <= 1 ? time_to_m : mean(time_to_m, dims=2)]
    #@assert length(o_mk_src) == length(o_mk_data)
    append!(src, [o_mk_src; o_mk_tar]) #bidirectional
    append!(tar, [o_mk_tar; o_mk_src])
    append!(edge_data, repeat(o_mk_data, 2))
  end
  m_k_src = repeat(num_operations+1:num_operations+M, s.K)
  m_k_tar = vcat([fill(num_operations + M + k, M) for k in 1:s.K]...)
  m_k_data = vcat([s.transport_time[s.assigned[s.last_o_k[k], 2], 1:M] for k in 1:s.K]...)
  #@assert length(m_k_src) == length(m_k_data) == length(m_k_tar)
  append!(src, [m_k_src; m_k_tar]) #bidirectional
  append!(tar, [m_k_tar; m_k_src])
  append!(edge_data, repeat(m_k_data, 2))
  #self loops
  append!(src, 1:num_nodes)
  append!(tar, 1:num_nodes)
  append!(edge_data, zeros(Float32, num_nodes))
  return GNNGraph(
    src,
    tar,
    num_nodes=num_nodes,
    ndata=node_data,
    edata=edge_data')
end

#####
##### Interaction APIc
#####

function GI.action_string(g::GameEnv, action)
  o, m, k = properties(g, action)
  return "o: " * string(o) * " m: " * string(m) * " k: " * string(k)
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

function print_graph(g::GameEnv)
  graph = GI.vectorize_state(GI.spec(g), g)
  is_scheduled = g.assigned[1:end-2, 2] .!= 0
  operation_nodes = ones(Int, length(is_scheduled))
  operation_nodes[is_scheduled] .= 2
  membership = [operation_nodes; 3; 4; fill(5, g.M + 1); fill(6, g.K)]
  nodecolor = [colorant"lightblue", colorant"blue", colorant"green", colorant"red", colorant"orange", colorant"yellow"]
  nodefillc = nodecolor[membership]
  draw(PNG("games/fjspt3/graphs/graph" * string(count(is_scheduled)) * ".png", 50cm, 50cm), gplot(graph, nodelabel=1:nv(graph), nodefillc=nodefillc))
end

function print_schedule(g::GameEnv, id; is_machine)
  print(crayon"white", is_machine ? "m" : "k", id, ":")
  line = []
  datarow = is_machine ? 2 : 1
  o = is_machine ? g.last_o_m[id] : g.last_o_k[id]
  t = g.done_time[g.job_ids[end]+1, datarow]
  m_i = g.M+1
  while (o != g.job_ids[end])
    i = count(<=(o), g.job_ids)
    #ready+1->done->ready reverse
    m_k = g.assigned[o, 2]
    t_setup = is_machine ? 0 : min(t - g.done_time[o, datarow], g.transport_time[m_k ,m_i])
    append!(line, [repeat("=", t_setup); crayon"dark_gray"])
    append!(line, [repeat("-", t - g.done_time[o, datarow] - t_setup); crayon"dark_gray"])
    append!(line, [repeat("=", g.done_time[o, datarow] - g.ready_time[o, datarow]); Crayon(foreground=i)])
    t = g.ready_time[o, datarow]
    is_first = o in g.job_ids
    prev_o_i = is_first ? g.job_ids[end] : o - 1
    m_i = g.assigned[prev_o_i, 2]
    o = maximum([(g.done_time[o, datarow], o) for o in filter(o -> g.assigned[o, datarow] == id && g.done_time[o, datarow] <= t, 1:(g.job_ids[end]-1))], init=(0, g.job_ids[end]))[2]
  end
  append!(line, [repeat("-", t); crayon"dark_gray"])
  print(reverse(line)...)
  println(crayon"reset")
end

function GI.render(g::GameEnv)
  for m in 1:g.M
    print_schedule(g, m, is_machine = true)
  end
  for k in 1:g.K
    print_schedule(g, k, is_machine = false)
  end
  for n in 1:g.N
    print(Crayon(foreground=n), "n", n, " ")
  end
  print(crayon"white", "makespan: ", g.done_time[g.job_ids[end]+1, 2] * 1)
  print(crayon"white", " reward: ", GI.white_reward(g))
  println(crayon"reset")
  #print_graph(g)
end
