using Optimisers
Network = NetLib.Gat

netparams = NetLib.GatHP()

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=PLSchedule([1, 10], [1000, 1000]),
    num_workers=PLSchedule([1, 10], [20, 20]),
    batch_size=PLSchedule([1, 10], [10, 10]),
    use_gpu=true,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=500,
    cpuct=1.0,
    temperature=PLSchedule([1, 5], [0.0, 0.0]),
    dirichlet_noise_ϵ=0.,
    dirichlet_noise_α=0.03))


learning = LearningParams(
  use_gpu=true,
  use_position_averaging=false,
  samples_weighing_policy=CONSTANT_WEIGHT,
  l2_regularization=1e-5,
  optimiser= Optimisers.Nesterov(1e-2, 0.9), 
  learnrate = PLSchedule([1, 2, 10, 60, 120], [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]), 
  batch_size=1_200,
  loss_computation_batch_size=4_500,
  nonvalidity_penalty=0.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=2_000,
  num_checkpoints=1,
  rewards_renormalization = 0.2)

benchmark_sim = SimParams(
  self_play.sim;
  num_games=PLSchedule([1, 10], [44, 44]),
  num_workers=ConstSchedule(20),
  batch_size=ConstSchedule(10),
  deterministic = true)

benchmark = [
  # Benchmark.Single(
  #   Benchmark.Full(MctsParams(self_play.mcts, temperature=ConstSchedule(0.),  dirichlet_noise_ϵ=0.)),
  #   benchmark_sim),
  Benchmark.Single(
    Benchmark.NetworkOnly(0.0),
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
  mem_buffer_size=PLSchedule(110_000))

experiment = Experiment(
  "fjspt3", GameSpec(PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [7, 7]), 
                    PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [7, 7]), 
                    PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [5, 5]), 
                    PLSchedule([1, 10], [2, 2])=>PLSchedule([1, 10], [2, 2]), 
                    1=>5,
                    1=>5), params, Network, netparams, benchmark)
