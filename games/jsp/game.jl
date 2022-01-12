
using CommonRLInterface
using StaticArrays
using Crayons
using Random

const RL = CommonRLInterface


const m = 3 #num machines
const n = 4 #num jobs
const p_min = 1#min time
const p_max = 5#max time

const num_sym = 16
const sqr_num_sym = round(sqrt(num_sym))

mutable struct World <: AbstractEnv
  process_time::MMatrix{n, m, UInt8} #Time of operations
  precendence::MMatrix{n, m, UInt8}#Order of operations
  machine_ready_time::MVector{m, UInt16}#Time before machine is ready
  job_ready_time::MVector{n, UInt16}#Time before job is ready
  c_min::UInt16
end

function World()
  process_time = rand(p_min:p_max,n,m)
  return World(
    process_time,
    reduce(hcat,[randperm(m) for i in 1:n])',
    zeros(UInt16,m),
    zeros(UInt16,n),
    maximum(sum(process_time[:,m] for m in 1:m)))
end

function RL.reset!(env::World) #Reset enviroment
  env.process_time = rand(p_min:p_max,n,m)
  env.precendence = reduce(hcat,[randperm(m) for i in 1:n])'
  env.machine_ready_time = zeros(Int,m)
  env.job_ready_time = zeros(Int,n)
  env.c_min = maximum(sum(env.process_time[:,m] for m in 1:m))
end  

RL.actions(env::World) = [i for i = 1:n] 

RL.observe(env::World) = [env.process_time, env.precendence, env.machine_ready_time, env.job_ready_time]

RL.terminated(env::World) = iszero(env.process_time)

function RL.act!(env::World, a) #Job a
  for i = 1:m
    b = env.precendence[a,i] #Machine b
    if(!iszero(b)) #If not done, schedule it
      done_time = max(env.machine_ready_time[b], env.job_ready_time[a]) + env.process_time[a,b]
      env.machine_ready_time[b] = done_time #Machine isn't usable untill scheduled operation is done
      env.job_ready_time[a] = done_time 
      env.process_time[a,b] = 0
      env.precendence[a,i] = 0
      break
    end
  end
  if(iszero(env.process_time))
    return env.c_min-convert(Float32,maximum(env.machine_ready_time))
  else
    return 0.0
  end  
end
@provide RL.valid_action_mask(env::World) = [!iszero(env.process_time[i,:]) for i = 1:n]
@provide RL.player(env::World) = 1 # An MDP is a one player game
@provide RL.players(env::World) = [1]
@provide RL.clone(env::World) = World(copy(env.process_time), copy(env.precendence), copy(env.machine_ready_time), copy(env.job_ready_time), env.c_min)
@provide RL.state(env::World) = [env.process_time, env.precendence, env.machine_ready_time, env.job_ready_time]

@provide function RL.setstate!(env::World, state)
  env.process_time, env.precendence, env.machine_ready_time, env.job_ready_time = state
  env.c_min = maximum(sum(env.process_time[:,m] for m in 1:m))
end

# Additional functions needed by AlphaZero.jl that are not present in 
# CommonRlInterface.jl. Here, we provide them by overriding some functions from
# GameInterface. An alternative would be to pass them as keyword arguments to
# CommonRLInterfaceWrapper.

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
function normalize(vector)
  min_t = minimum(vector)
  max_t = maximum(vector)
  return (vector.-min_t)./(max_t-min_t+1e-8)
end

#normalized state vector
function GI.vectorize_state(env::World, state) #[env.process_time, env.precendence, env.machine_ready_time, env.job_ready_time]
  precendence_state = falses(m, m, n)
  for i = 1:m
    for j = 1:n
      if(!iszero(state[2][j,i]))
        precendence_state[i, state[2][j,i], j] = true
      end
    end
  end
  return [state[1]./p_max..., precendence_state..., normalize([state[3];state[4]])...]
end

function GI.action_string(env::World, a)
  return string(a)
end

# function GI.parse_action(env::World, s)

# end

# function GI.read_state(env::World)

# end

GI.heuristic_value(::World) = 0.

GameSpec() = CommonRLInterfaceWrapper.Spec(World())