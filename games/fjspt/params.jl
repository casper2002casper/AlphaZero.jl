Network = NetLib.Gin

netparams = NetLib.GinHP()

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=PLSchedule([1, 10], [3000, 3000]),
    num_workers=PLSchedule([1, 10], [150, 150]),
    batch_size=PLSchedule([1, 10], [50, 50]),
    use_gpu=true,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=600,
    cpuct=1.0,
    adaptive_cpuct = false,
    temperature=ConstSchedule(0.05),
    dirichlet_noise_ϵ=0.,
    dirichlet_noise_α=0.2))


learning = LearningParams(
  use_gpu=true,
  use_position_averaging=false,
  samples_weighing_policy=CONSTANT_WEIGHT,
  l2_regularization=1e-5,
  optimiser=Adam(lr=1e-3),
  batch_size=1500,
  loss_computation_batch_size=2048,
  nonvalidity_penalty=0.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=2_000,
  num_checkpoints=1,
  rewards_renormalization = 0.2)

benchmark_sim = SimParams(
  self_play.sim;
  num_games=PLSchedule([1, 10], [44, 44]),
  num_workers=ConstSchedule(32),
  batch_size=ConstSchedule(16),
  deterministic = true)

benchmark = [
  Benchmark.Single(
    Benchmark.Full(MctsParams(self_play.mcts, temperature=ConstSchedule(0.))),
    benchmark_sim),
  Benchmark.Single(
    Benchmark.NetworkOnly(),
    benchmark_sim)]

  arena = ArenaParams(
  sim=benchmark_sim,
  mcts=MctsParams(self_play.mcts, temperature=ConstSchedule(0.)),
  update_threshold=0.01)

params = Params(
  arena=nothing,
  self_play=self_play,
  learning=learning,
  num_iters=500,
  memory_analysis=nothing,
  ternary_rewards=false,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(300_000))

experiment = Experiment(
  "fjspt", GameSpec(PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [7, 7]), 
                    PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [7, 7]), 
                    PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [5, 5]), 
                    PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [2, 2]), 
                    1=>5,
                    1=>5), params, Network, netparams, benchmark)
