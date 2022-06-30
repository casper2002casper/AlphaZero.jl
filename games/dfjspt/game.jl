include("../fjspt/game.jl")

function GI.disturbe!(spec::GameSpec, g::GameEnv, rng::AbstractRNG, itc)
  if (rand(rng) < 1 / 10)
    type = rand(rng, 1:6)
    if (type == 1)#Job arrival
      new_opperations = rand(rng, 1:M)
      for i in 1:new_opperations
        times = rand(rng, spec.P.first:spec.P.second, g.M)
        alts = rand(rng, spec.A.first[itc]:spec.A.second[itc])
        ind = randperm(rng, g.M)[1:g.M-alts]
        times[ind] .= 0xff
        g.p_time = [g.p_time; times]
      end
      count = 0
      num_operations = [new_opperations]
      for i in 1:g.N_OPP
        if(g.conj_tar[i*2]>g.N_OPP*2)
          append!(num_operations, count)
          count = 0
        else
          count += 1
        end
      end
      g.N += 1
      g.N_OPP += new_opperations
      g.T += new_opperations * 2 + 1
      g.S += new_opperations * 2 + 1
      g.conj_tar = generate_conjuctive_edges(num_operations, g.N_OPP, g.T, g.S)
      job_done_time = gen_done_time(g.p_time, g.t_time, g.conj_tar, [g.S], g.S, g.T)
      g.done_time = max.(g.done_time, job_done_time)
      g.prev_operation = [[S (M+1)] g.prev_operation]
      g.adaptive_nodes = gen_action_values(g.process_time, g.transport_time, g.conj_tar, [S; g.prev_operation[:,1]], g.prev_operation, g.prev_machine, g.prev_vehicle, g.done_time, g.K, g.S)
      g.UB = sum(maximum(replace(g.p_time, 0xff => 0x00), dims=2)) + maximum(g.t_time) * (2 * g.N_OPP + g.N)
      g.LB = g.done_time[g.T]
      g.conj_src = [collect(1:g.N_OPP*2+g.N); g.T; g.S * ones(g.N)]
      g.is_done = [falses(new_opperations*2); g.is_done[1:end-g.N-1]; false; g.is_done[end-g.N+2:end] false; true]
    elseif (type == 2)#Job cancelation
      removed_job = rand(rng, 1:g.N)
      first_opp = g.conj_tar[g.S + removed_job - 1]÷2
      last_opp = findfirst(==(g.T - removed_job), g.conj_tar)÷2
      num_opps = last_opp - first_opp
      bit_map = [trues(first_opp*2); falses(num_opps*2); trues(g.T - removed_job - last_opp*2); false; trues(g.N); false; trues(g.N - removed_job)]
      g.process_time = g.process_time[bitmap[1:2:2*g.N_OPP]]
      g.conj_tar = g.conj_tar[bit_map]
      g.is_done = g.is_done[bitmap[1:g.S]]
      g.done_time = g.done_time[bitmap[1:g.S]]
      g.done_time[g.T] = maximum(g.done_time[g.N_OPP*2+1:g.N_OPP*2+g.N-1])
      g.prev_operation = [g.prev_operation[1:removed_job-1, :]; g.prev_operation[removed_job+1:end,:]]
      g.N -= 1
      g.N_OPP -= num_opps
      g.T -= num_opps*2
      g.S -= num_opps*2
      g.conj_src = [collect(1:g.N_OPP*2+g.N); g.T; g.S * ones(g.N)]
      disjunctive_mask = (g.disj_src .> (first_opp*2) .&& g.disj_src .< (last_opp*2)) .&& (g.disj_tar .> (first_opp*2) .&& g.disj_tar .< (last_opp*2))
      g.disj_src = g.disj_src[!disjunctive_mask]
      g.disj_tar = g.disj_tar[!disjunctive_mask]
      g.adaptive_nodes = gen_action_values(g.process_time, g.transport_time, g.conj_tar, g.prev_operation[:,1], g.prev_operation, g.prev_machine, g.prev_vehicle, g.done_time, g.K, g.S)
      g.UB = sum(maximum(replace(g.p_time, 0xff => 0x00), dims=2)) + maximum(g.t_time) * (2 * g.N_OPP + g.N)
      g.LB = g.done_time[g.T]
    elseif (type == 3)#Machine breakdown

    elseif (type == 4)#Machine repair
      g.M
      g.process_time
      g.transport_time
    elseif (type == 5)#Process time change
      g.process_time
    elseif (type == 6)#Transport time change  
      g.transport_time
    else

    end
    return true
  end
  return false
end