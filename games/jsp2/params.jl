Network = NetLib.Gnn

netparams = NetLib.GnnHP()

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=300,
    num_workers=16,
    batch_size=8,
    use_gpu=false,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=40,
    cpuct=1.4,
    adaptive_cpuct = true,
    temperature=ConstSchedule(0.),
    dirichlet_noise_ϵ=0.,
    dirichlet_noise_α=1.))


learning = LearningParams(
  use_gpu=false,
  use_position_averaging=false,
  samples_weighing_policy=CONSTANT_WEIGHT,
  l2_regularization=1e-4,
  optimiser=Adam(lr=5e-3),
  batch_size=64,
  loss_computation_batch_size=64,
  nonvalidity_penalty=0.1,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=5_000,
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
  mem_buffer_size=PLSchedule(5_000))

benchmark_sim = SimParams(
  self_play.sim;
  num_games=100,
  num_workers=16,
  batch_size=8)

  benchmark = [
    Benchmark.Single(
      Benchmark.Full(self_play.mcts),
      benchmark_sim),
    Benchmark.Single(
      Benchmark.NetworkOnly(),
      benchmark_sim)]

experiment = Experiment(
  "jsp", GameSpec(), params, Network, netparams, benchmark)