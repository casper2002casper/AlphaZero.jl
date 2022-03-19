Network = NetLib.Gin

netparams = NetLib.GinHP()

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=1500,
    num_workers=128,
    batch_size=32,
    num_workers=128,
    batch_size=32,
    use_gpu=true,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=300,
    cpuct=0.8,
    adaptive_cpuct = true,
    temperature=ConstSchedule(0.05),
    dirichlet_noise_ϵ=0.,
    dirichlet_noise_α=0.2))


learning = LearningParams(
  use_gpu=true,
  use_position_averaging=false,
  samples_weighing_policy=LOG_WEIGHT,
  l2_regularization=1e-5,
  optimiser=Adam(lr=1e-4),
  batch_size=2048,
  loss_computation_batch_size=2048,
  nonvalidity_penalty=0.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=2_000,
  num_checkpoints=1,
  rewards_renormalization = 22)

params = Params(
  arena=nothing,
  self_play=self_play,
  learning=learning,
  num_iters=50,
  memory_analysis=nothing,
  ternary_rewards=false,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(300_000))

benchmark_sim = SimParams(
  self_play.sim;
  num_games=24,
  num_workers=24,
  batch_size=12,
  deterministic = true)

benchmark = [
  Benchmark.Single(
    Benchmark.Full(MctsParams(self_play.mcts, temperature=ConstSchedule(0.))),
    benchmark_sim),
  Benchmark.Single(
    Benchmark.NetworkOnly(),
    benchmark_sim)]

experiment = Experiment(
  "jsp", GameSpec(), params, Network, netparams, benchmark)