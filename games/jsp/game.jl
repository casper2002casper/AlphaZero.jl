
using CommonRLInterface
using StaticArrays
using Crayons
using Random
using GraphNeuralNetworks

const RL = CommonRLInterface


const M = 3 #num machines
const N = 4 #num jobs
const P_MIN = 1#min time
const P_MAX = 5#max time
const NUM_N = M*N + 2
const S = M*N+1
const T = M*N+N+1

const num_sym = 16
const sqr_num_sym = round(sqrt(num_sym))

nm2i(n,m) = n*N + m 

mutable struct World <: AbstractEnv
  #Problem instance
  process_time::SMatrix{M, N, UInt8} #Time of operations
  conj_src::SVector{NUM_N, UInt8}
  conj_tar::SVector{NUM_N, UInt8}
  #State
  disj_src::Vector{UInt8}
  disj_tar::Vector{UInt8}
  is_done::MMatrix{M, N, Bool}
  done_time::MMatrix{M, N, UInt16}
end

function generate_conjuctive_edges()
  order = reduce(vcat,[(randperm(M).+i*N) for i in 0:N-1])
  target = zeros(UInt8, N*(M+1))
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

function World()
  process_time = rand(P_MIN:P_MAX,N,M)
  return World(
    rand(P_MIN:P_MAX,N,M),
    [collect(1:N*M)..., S * ones(N)..., T],
    generate_conjuctive_edges(),
    [],
    [],
    falses(M, N),
    zeros(UInt16, M, N)
    )
end

function RL.reset!(env::World) #Generate new problem
  env.process_time = rand(P_MIN:P_MAX,N,M)
  env.conj_tar = generate_conjuctive_edges()
  env.disj_src = []
  env.disj_tar = []
  env.done_time = zeros(UInt16,M, N)
end  

RL.actions(env::World) = [i for i = 1:NUM_N] 

RL.observe(env::World) = env

RL.terminated(env::World) = all(env.is_done)

function RL.act!(env::World, a) #Job a


  for i = 1:M
    b = env.precendence[a,i] #Machine b
    if(!iszero(b)) #If not done, schedule it
      #job_ready_time = completion_time last operation n, m-1
      #machine_ready_time = completion_time last on machine max(n, m)
      done_time = max(env.machine_ready_time[b], env.job_ready_time[a]) + env.process_time[a,b]
      env.machine_ready_time[b] = done_time #Machine isn't usable untill scheduled operation is done
      env.job_ready_time[a] = done_time 
      env.process_time[a,b] = 0
      env.precendence[a,i] = 0
      break
    end
  end
  if(iszero(env.process_time))
    return -convert(Float32,maximum(env.machine_ready_time))
  else
    return 0.0
  end  
end
@provide function RL.valid_action_mask(env::World) 
  valid_actions = zeros(Bool, NUM_N)
  for (n,m) in enumerate(env.to_schedule)
    valid_actions[nm2i(n,m)] = true
  end
  return valid_actions
end
@provide RL.player(env::World) = 1 # An MDP is a one player game
@provide RL.players(env::World) = [1]
@provide RL.clone(env::World) = #World(copy(env.process_time), copy(env.precendence), copy(env.machine_ready_time), copy(env.job_ready_time))
@provide RL.state(env::World) = env

@provide function RL.setstate!(env::World, state)
  env.process_time, env.precendence, env.machine_ready_time, env.job_ready_time = state
  env.c_min = maximum(sum(env.process_time[:,m] for m in 1:M))
end

function GI.vectorize_state(env::World, state) #[env.process_time, env.precendence, env.machine_ready_time, env.job_ready_time]
  return GNNGraph([env.conj_src;env.disj_src], 
                  [env.conj_tar; env.conj_tar], 
                  [env.process_time..., ones(UInt8, size(env.disj_tar))], 
                  num_nodes = NUM_N, 
                  ndata = [env.done_time...])
end

function GI.action_string(env::World, a)
  return string(a)
end

# function GI.symmetries(env::World, state) 
#   #[masked_process_time, precendence_state, normalized_machine_ready_time, normalized_job_ready_time, env.time]
#   return [([state[1][order_n, order_m], 
#             state[2][order_m, order_m, order_n],
#             state[3][order_m],
#             state[4][order_n],
#             state[5]],
#             order_n) 
#             for order_n in unique([randperm(n) for _ in 1:sqr_num_sym])
#               for order_m in unique([randperm(m) for () in 1:sqr_num_sym])]
#   end

# function GI.render(env::World)
# end


# function GI.parse_action(env::World, s)
# end

# function GI.read_state(env::World)
# end

GI.heuristic_value(::World) = 0.

GameSpec() = CommonRLInterfaceWrapper.Spec(World())