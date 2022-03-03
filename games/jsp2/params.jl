Network = NetLib.Gin

netparams = NetLib.GinHP()

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=64,
    num_workers=24,
    batch_size=12,
    use_gpu=true,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=300,
    cpuct=1.4,
    adaptive_cpuct = false,
    temperature=ConstSchedule(1.),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=10/7))


learning = LearningParams(
  use_gpu=true,
  use_position_averaging=false,
  samples_weighing_policy=CONSTANT_WEIGHT,
  l2_regularization=1e-4,
  optimiser=Adam(lr=5e-3),
  batch_size=812,
  loss_computation_batch_size=2048,
  nonvalidity_penalty=0.,
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
  mem_buffer_size=PLSchedule(10_000))

benchmark_sim = SimParams(
  self_play.sim;
  num_games=12,
  num_workers=1,
  batch_size=1,
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