Network = NetLib.Gin

netparams = NetLib.GinHP()

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=32,
    num_workers=8,
    batch_size=4,
    use_gpu=true,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=400,
    cpuct=1.4,
    adaptive_cpuct = false,
    temperature=ConstSchedule(0.),
    dirichlet_noise_ϵ=0.,
    dirichlet_noise_α=1.))


learning = LearningParams(
  use_gpu=true,
  use_position_averaging=false,
  samples_weighing_policy=CONSTANT_WEIGHT,
  l2_regularization=1e-4,
  optimiser=Adam(lr=5e-3),
  batch_size=64,
  loss_computation_batch_size=64,
  nonvalidity_penalty=0.1,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=2_000,
  num_checkpoints=1,
  rewards_renormalization = 1)


params = Params(
  arena=nothing,
  self_play=self_play,
  learning=learning,
  num_iters=10,
  memory_analysis=nothing,
  ternary_rewards=false,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(2_500))

benchmark_sim = SimParams(
  self_play.sim;
  num_games=12,
  num_workers=4,
  batch_size=2)

  benchmark = [
    Benchmark.Single(
      Benchmark.Full(self_play.mcts),
      benchmark_sim),
    Benchmark.Single(
      Benchmark.NetworkOnly(),
      benchmark_sim)]

experiment = Experiment(
  "jsp", GameSpec(), params, Network, netparams, benchmark)