using MLUtils

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

  if wp == CONSTANT_WEIGHT
    w = Float32[1]
  elseif wp == LOG_WEIGHT
    w = Float32[log2(e.n) + 1]
  else
    @assert wp == LINEAR_WEIGHT
    w = Float32[n]
  end
  x = GI.vectorize_state(gspec, e.s)
  #a = GI.actions_mask(GI.init(gspec, e.s))
  p = Float32.(e.π)
  v = [e.z]
  return (; w, x, p, v)
end

function convert_samples(
    gspec::AbstractGameSpec,
    wp::SamplesWeighingPolicy,
    es::AbstractVector{<:TrainingSample})
  ces = [convert_sample(gspec, wp, e) for e in es]
  W = MLUtils.batch([e.w for e in ces])
  X = typeof(ces[1].x) <: Matrix ? MLUtils.batch([e.x for e in ces]) : [e.x for e in ces] 
  #A = [e.a for e in ces] 
  P = MLUtils.batch([e.p for e in ces], 0)
  V = MLUtils.batch([e.v for e in ces])
  function f32(arr)
    if typeof(arr) <: Matrix
      return convert(AbstractArray{Float32}, arr)
    else
      return arr
    end
  end
  return map(f32, (; W, X, P, V))
end

function MLUtils.batch(xs::AbstractVector{<:AbstractVector}, pad; n=maximum(length(x) for x in xs))
  return MLUtils.batch(rpad.(xs, n, pad))
end

#####
##### Loss Function
#####

# Surprisingly, Flux does not like the following code (scalar operations):
# mse_wmean(ŷ, y, w) = sum((ŷ .- y).^2 .* w) / sum(w)
mse_wmean(ŷ, y, w) = sum(((ŷ .- y)./y).^2 .* w) / sum(w)

klloss_wmean(π̂, π, w) = -sum(π .* log.(π̂ .+ eps(eltype(π))) .* w) / sum(w)

entropy_wmean(π, w) = -sum(π .* log.(π .+ eps(eltype(π))) .* w) / sum(w)

wmean(x, w) = sum(x .* w) / sum(w)

function losses(nn, regws, params, Wmean, Hp, (W, X, P, V))
  # `regws` must be equal to `Network.regularized_params(nn)`
  creg = params.l2_regularization
  cinv = params.nonvalidity_penalty
  renorm = params.rewards_renormalization
  P̂, V̂ = Network.forward(nn, X)
  P̂ = MLUtils.batch(P̂, 0, n=size(P,1))
  Lp = klloss_wmean(P̂, P, W) - Hp
  Lv = mse_wmean(V̂, V, W)
  Lreg = iszero(creg) ?
    zero(Lv) :
    creg * sum(sum(w .* w) for w in regws)
  # Linv = iszero(cinv) ?
  #   zero(Lv) :
  #   cinv * wmean(p_invalid, W)
  L = (mean(W) / Wmean) * (Lp + Lv + Lreg)
  return (L, Lp, Lv, Lreg, 0)
end

#####
##### Trainer Utility
#####

struct Trainer
  network :: AbstractNetwork
  samples :: AbstractVector{<:TrainingSample}
  params :: LearningParams
  dataloader :: Flux.Data.DataLoader # (W, X, A, P, V) tuple obtained after converting `samples`
  Wmean :: Float32
  Hp :: Float32
  function Trainer(gspec, network, samples, params; test_mode=false)
    if params.use_position_averaging
      samples = merge_by_state(samples)
    end
    data = convert_samples(gspec, params.samples_weighing_policy, samples)
    network = Network.copy(network, on_gpu=params.use_gpu, test_mode=test_mode)
    W, X, P, V = data
    Wmean = mean(W)
    Hp = entropy_wmean(P, W)
    # Create a batches stream
    batchsize = min(params.batch_size, length(W))
    dataloader = MLUtils.DataLoader(data; batchsize, partial=false, shuffle=true)
    return new(network, samples, params, dataloader, Wmean, Hp) 
  end
end

data_weights(tr::Trainer) = tr.dataloader.data.W

num_samples(tr::Trainer) = length(data_weights(tr))

num_batches_total(tr::Trainer) = length(tr.dataloader)

function get_trained_network(tr::Trainer)
  return Network.copy(tr.network, on_gpu=false, test_mode=true)
end

function batch_updates!(tr::Trainer, n)
  regws = Network.regularized_params(tr.network)
  L(batch...) = losses(tr.network, regws, tr.params, tr.Wmean, tr.Hp, batch)[1]
  ls = Vector{Float32}()
  Network.train!(tr.network, tr.params.optimiser, L, tr.dataloader, n) do i, l
    push!(ls, l)
  end
  Network.gc(tr.network)
  return ls
end

#####
##### Generating debugging reports
#####

function mean_learning_status(reports, ws)
  L     = wmean([r.loss.L     for r in reports], ws)
  Lp    = wmean([r.loss.Lp    for r in reports], ws)
  Lv    = wmean([r.loss.Lv    for r in reports], ws)
  Lreg  = wmean([r.loss.Lreg  for r in reports], ws)
  Linv  = wmean([r.loss.Linv  for r in reports], ws)
  Hpnet = wmean([r.Hpnet      for r in reports], ws)
  Hp    = wmean([r.Hp         for r in reports], ws)
  return Report.LearningStatus(Report.Loss(L, Lp, Lv, Lreg, Linv), Hp, Hpnet)
end

function learning_status(tr::Trainer, samples)
  # As done now, this is slighly inefficient as we solve the
  # same neural network inference problem twice
  samples = Network.convert_input_tuple(tr.network, samples)
  W, X, P, V = samples
  regws = Network.regularized_params(tr.network)
  Ls = losses(tr.network, regws, tr.params, tr.Wmean, tr.Hp, samples)
  Ls = Network.convert_output_tuple(tr.network, Ls)
  Pnet, _ = Network.forward(tr.network, X)
  Pnet = batch(Pnet, 0)
  Hpnet = entropy_wmean(Pnet, W)
  Hpnet = Network.convert_output(tr.network, Hpnet)
  return Report.LearningStatus(Report.Loss(Ls...), tr.Hp, Hpnet)
end

function learning_status(tr::Trainer)
  batchsize = min(tr.params.loss_computation_batch_size, num_samples(tr))
  batches = MLUtils.DataLoader(tr.dataloader.data; batchsize, partial=true)
  reports = []
  ws = []
  for batch in batches
    push!(reports, learning_status(tr, batch))
    push!(ws, sum(batch.W))
    #GC.gc(true)
  end
  return mean_learning_status(reports, ws)
end

function samples_report(tr::Trainer)
  status = learning_status(tr)
  # Samples in `tr.samples` can be merged by board or not
  num_samples = sum(e.n for e in tr.samples)
  num_boards = length(merge_by_state(tr.samples))
  Wtot = sum(data_weights(tr))
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
    map(stages) do es
      ts = [e.t for e in es]
      stats = samples_report(Tr(es))
      Report.StageSamples(minimum(ts), maximum(ts), stats)
    end
  end
  return Report.Memory(latest_batch, all_samples, per_game_stage)
end
