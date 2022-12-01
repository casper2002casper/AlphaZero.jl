Network = NetLib.Gin

netparams = NetLib.GinHP()

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=PLSchedule([1, 10], [2000, 2000]),
    num_workers=PLSchedule([1, 10], [20, 20]),
    batch_size=PLSchedule([1, 10], [10, 10]),
    use_gpu=true,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=300,
    cpuct=1.3,
    temperature=PLSchedule([1, 10], [1.0, 0.0]),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=0.05))


learning = LearningParams(
  use_gpu=true,
  use_position_averaging=false,
  samples_weighing_policy=CONSTANT_WEIGHT,
  l2_regularization=1e-5,
  optimiser=Adam(lr=5e-4),
  learnrate=ConstSchedule(1e-3),
  batch_size=3250,
  loss_computation_batch_size=1500,
  nonvalidity_penalty=0.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=2_000,
  num_checkpoints=1,
  rewards_renormalization = 0.2)

benchmark_sim = SimParams(
  self_play.sim;
  num_games=PLSchedule([1, 10], [44, 44]),
  num_workers=ConstSchedule(31),
  batch_size=ConstSchedule(16),
  deterministic = true)

benchmark = [
  # Benchmark.Single(
  #   Benchmark.Full(MctsParams(self_play.mcts, temperature=ConstSchedule(0.),  dirichlet_noise_ϵ=0.)),
  #   benchmark_sim),
  Benchmark.Single(
    Benchmark.NetworkOnly(),
    benchmark_sim)]

  arena = ArenaParams(
  sim=benchmark_sim,
  mcts=MctsParams(self_play.mcts, temperature=ConstSchedule(0.),  dirichlet_noise_ϵ=0.),
  update_threshold=0.01)

params = Params(
  arena=nothing,
  self_play=self_play,
  learning=learning,
  num_iters=500,
  memory_analysis=nothing,
  ternary_rewards=false,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(1_000_000))

experiment = Experiment(
  "fjspt2", GameSpec(PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [7, 7]), 
                    PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [7, 7]), 
                    PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [5, 5]), 
                    PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [2, 2]), 
                    1=>5,
                    1=>5), params, Network, netparams, benchmark)
