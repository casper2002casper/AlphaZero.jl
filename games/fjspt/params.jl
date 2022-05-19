Network = NetLib.Gin

netparams = NetLib.GinHP()

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=PLSchedule([1, 10], [1000, 360]),
    num_workers=PLSchedule([1, 10], [500, 360]),
    batch_size=PLSchedule([1, 10], [250, 180]),
    use_gpu=true,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=300,
    cpuct=0.8,
    adaptive_cpuct = false,
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
  rewards_renormalization = 1)

params = Params(
  arena=nothing,
  self_play=self_play,
  learning=learning,
  num_iters=50,
  memory_analysis=nothing,
  ternary_rewards=false,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(500_000))

benchmark_sim = SimParams(
  self_play.sim;
  num_games=PLSchedule([1, 10], [64, 12]),
  num_workers=ConstSchedule(44),
  batch_size=ConstSchedule(22),
  deterministic = true)

benchmark = [
  Benchmark.Single(
    Benchmark.Full(MctsParams(self_play.mcts, temperature=ConstSchedule(0.))),
    benchmark_sim),
  Benchmark.Single(
    Benchmark.NetworkOnly(),
    benchmark_sim)]

experiment = Experiment(
  "fjspt", GameSpec(PLSchedule([1, 10], [4, 4])=>PLSchedule([1, 10], [4, 4]), 
                    PLSchedule([1, 10], [4, 4])=>PLSchedule([1, 10], [4, 4]), 
                    PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [2, 2]), 
                    PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [2, 2]), 
                    1=>5,
                    1=>5), params, Network, netparams, benchmark)
