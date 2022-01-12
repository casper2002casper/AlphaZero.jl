Network = NetLib.SimpleNet

netparams = NetLib.SimpleNetHP(
  width=50,
  depth_common=4,
  use_batch_norm=false)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=512,
    num_workers=16,
    batch_size=8,
    use_gpu=false,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=100,
    cpuct=1.4,
    adaptive_cpuct = true,
    temperature=ConstSchedule(0.),
    dirichlet_noise_ϵ=0.,
    dirichlet_noise_α=1.))


learning = LearningParams(
  use_gpu=false,
  use_position_averaging=true,
  samples_weighing_policy=CONSTANT_WEIGHT,
  l2_regularization=1e-4,
  optimiser=Adam(lr=5e-3),
  batch_size=64,
  loss_computation_batch_size=64,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=5_000,
  num_checkpoints=1)

  arena = ArenaParams(
    sim=SimParams(
      num_games=2048,
      num_workers=32,
      batch_size=16,
      use_gpu=false,
      reset_every=1,
      flip_probability=0.,
      alternate_colors=true),
    mcts = self_play.mcts,
    update_threshold=0.00)

params = Params(
  arena=nothing,
  self_play=self_play,
  learning=learning,
  num_iters=10,
  memory_analysis=nothing,
  ternary_rewards=false,
  use_symmetries=true,
  mem_buffer_size=PLSchedule(80_000))

benchmark_sim = SimParams(
  arena.sim;
  num_games=2048,
  num_workers=16,
  batch_size=8)

  benchmark = [
    # Benchmark.Single(
    #   Benchmark.Full(self_play.mcts),
    #   benchmark_sim)]#,
    Benchmark.Single(
      Benchmark.NetworkOnly(),
      benchmark_sim)]

experiment = Experiment(
  "jsp", GameSpec(), params, Network, netparams, benchmark)