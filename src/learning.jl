using MLUtils, Distributed

#####
##### Converting samples
#####

# A samples collection is represented on the learning side as a (W, X, A, P, V)
# named-tuple. Each component is a `Float32` tensor whose last dimension corresponds
# to the sample index. Writing `n` the number of samples and `a` the total
# number of actions:
# - W (size 1×n) contains the samples weights
# - X (size …×n) contains the board representations
# - A (size a×n) contains the action masks (values are either 0 or 1)
# - P (size a×n) contains the recorded MCTS policies
# - V (size 1×n) contains the recorded values
# Note that the weight of a sample is computed as an increasing
# function of its `n` field.

function convert_sample(
    gspec::AbstractGameSpec,
    wp::SamplesWeighingPolicy,
    e::TrainingSample)
  x = GI.vectorize_state(gspec, e.s)
  p = Float32.(e.π)
  v = [e.z]
  return (; x, p, v)
end

function convert_samples(
    gspec::AbstractGameSpec,
    wp::SamplesWeighingPolicy,
    es::AbstractVector{<:TrainingSample})
  ces = [convert_sample(gspec, wp, e) for e in es]
  X = typeof(ces[1].x) <: Matrix ? MLUtils.batch([e.x for e in ces]) : [e.x for e in ces] 
  P = MLUtils.batch([e.p for e in ces], 0)
  V = MLUtils.batch([e.v for e in ces])
  function f32(arr)
    if typeof(arr) <: Matrix
      return convert(AbstractArray{Float32}, arr)
    else
      return arr
    end
  end
  return map(f32, (; X, P, V))
end

function MLUtils.batch(xs::AbstractVector{<:AbstractVector}, pad; n=maximum(length(x) for x in xs))
  return MLUtils.batch(rpad.(xs, n, pad))
end

#####
##### Loss Function
#####

# Surprisingly, Flux does not like the following code (scalar operations):
# mse_wmean(ŷ, y, w) = sum((ŷ .- y).^2 .* w) / sum(w)
mse_wmean(ŷ, y) = sum(((ŷ .- y)./y).^2) / length(y)

klloss_wmean(π̂, π) = -sum(π .* log.(π̂ .+ eps(eltype(π)))) / size(π, 2) 

entropy_wmean(π) = klloss_wmean(π, π)

function losses(nn, regws, params, Hp, (P, V, X); HPnet = false)
  # `regws` must be equal to `Network.regularized_params(nn)`
  creg = params.l2_regularization
  P̂, V̂ = Network.forward(nn, X)
  P̂ = MLUtils.batch(P̂, 0, n=size(P,1))
  Lp = klloss_wmean(P̂, P) - Hp
  Lv = mse_wmean(V̂, V)
  Lreg = iszero(creg) ?
    zero(Lv) :
    creg * sum(sum(w .* w) for w in regws)
  L = Lp + Lv + Lreg
  Hpnet = 0.0
  if(HPnet)
    Hpnet = entropy_wmean(P̂)
  end
  return (L, Lp, Lv, Lreg, 0, Hpnet)
end

#####
##### Trainer Utility
#####

struct Trainer
  network :: AbstractNetwork
  samples :: AbstractVector{<:TrainingSample}
  params :: LearningParams
  dataloader :: MLUtils.DataLoader # (W, X, A, P, V) tuple obtained after converting `samples`
  Hp :: Float32
  function Trainer(gspec, network, samples, params; test_mode=false)
    if params.use_position_averaging
      samples = merge_by_state(samples)
    end
    data = convert_samples(gspec, params.samples_weighing_policy, samples)
    network = Network.copy(network, on_gpu=params.use_gpu, test_mode=test_mode)
    X, P, V = data
    Hp = entropy_wmean(P)
    # Create a batches stream
    batchsize = min(params.batch_size, length(V))
    dataloader = MLUtils.DataLoader(data; batchsize, partial=false, shuffle=true, collate=true)
    return new(network, samples, params, dataloader, Hp) 
  end
end

num_samples(tr::Trainer) = length(tr.dataloader.data)

num_batches_total(tr::Trainer) = length(tr.dataloader)

function get_trained_network(tr::Trainer)
  return Network.copy(tr.network, on_gpu=false, test_mode=true)
end

function batch_updates!(tr::Trainer, n, itc)
  regws = Network.regularized_params(tr.network)
  L(batch...) = losses(tr.network, regws, tr.params, tr.Hp, batch)[1]
  ls = Vector{Float32}()
  Network.train!(tr.network, tr.params.optimiser, L, tr.dataloader, n, itc) do i, l
    push!(ls, l)
  end
  Network.gc(tr.network)
  return ls
end

#####
##### Generating debugging reports
#####

function mean_learning_status(reports)
  L     = mean(r.loss.L     for r in reports)
  Lp    = mean(r.loss.Lp    for r in reports)
  Lv    = mean(r.loss.Lv    for r in reports)
  Lreg  = mean(r.loss.Lreg  for r in reports)
  Linv  = mean(r.loss.Linv  for r in reports)
  Hpnet = mean(r.Hpnet      for r in reports)
  Hp    = mean(r.Hp         for r in reports)
  return Report.LearningStatus(Report.Loss(L, Lp, Lv, Lreg, Linv), Hp, Hpnet)
end

function learning_status(tr::Trainer, samples)
  samples = Network.convert_input_tuple(tr.network, samples)
  regws = Network.regularized_params(tr.network)
  Ls = losses(tr.network, regws, tr.params, tr.Hp, samples, HPnet = true)
  Ls = Network.convert_output_tuple(tr.network, Ls)
  return Report.LearningStatus(Report.Loss(Ls[1:5]...), tr.Hp, Ls[end])
end

function learning_status(tr::Trainer)
  batchsize = min(tr.params.loss_computation_batch_size, num_samples(tr))
  batches = MLUtils.DataLoader(tr.dataloader.data; batchsize, partial=true, collate=true)
  reports = []
  GC.gc(true)
  CUDA.memory_status()
  pool = default_worker_pool()
  for batch in batches
    l = remotecall(x->learning_status(tr, x), pool, batch)
    push!(reports, l)
    #CUDA.memory_status()
    #GC.gc(true)
  end
  reports = fetch.(reports)
  Distributed.@everywhere CUDA.reclaim()
  Distributed.@everywhere GC.gc(true)
  Distributed.@everywhere CUDA.reclaim()
  Distributed.@everywhere GC.gc(true)
  return mean_learning_status(reports)
end

function samples_report(tr::Trainer)
  status = learning_status(tr)
  # Samples in `tr.samples` can be merged by board or not
  num_samples = length(tr.samples)
  num_boards = length(merge_by_state(tr.samples))
  Wtot = num_samples
  return Report.Samples(num_samples, num_boards, Wtot, status)
end

function memory_report(
    mem::MemoryBuffer,
    nn::AbstractNetwork,
    learning_params::LearningParams,
    params::MemAnalysisParams
  )
  # It is important to load the neural network in test mode so as to not
  # overwrite the batch norm statistics based on biased data.
  Tr(samples) = Trainer(mem.gspec, nn, samples, learning_params, test_mode=true)
  all_samples = samples_report(Tr(get_experience(mem)))
  latest_batch = isempty(last_batch(mem)) ?
    all_samples :
    samples_report(Tr(last_batch(mem)))
  per_game_stage = begin
    es = get_experience(mem)
    sort!(es, by=(e->e.t))
    csize = ceil(Int, length(es) / params.num_game_stages)
    stages = collect(Iterators.partition(es, csize))
    # map(stages) do es
    #   ts = [e.t for e in es]
    #   stats = samples_report(Tr(es))
    #   Report.StageSamples(minimum(ts), maximum(ts), stats)
    # end
  end
  return Report.Memory(latest_batch, all_samples, per_game_stage)
end
