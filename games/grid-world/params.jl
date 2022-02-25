Network = NetLib.Gcn

netparams = NetLib.GcnHP()

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=2000,
    num_workers=8,
    batch_size=4,
    use_gpu=false,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=50,
    cpuct=sqrt(2),
    adaptive_cpuct=false,
    temperature=ConstSchedule(0.),
    dirichlet_noise_ϵ=0.,
    dirichlet_noise_α=1.))

arena = ArenaParams(
  sim=SimParams(
    num_games=100,
    num_workers=10,
    batch_size=10,
    use_gpu=false,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=true),
  mcts = self_play.mcts,
  update_threshold=0.00)

learning = LearningParams(
  use_gpu=false,
  use_position_averaging=true,
  samples_weighing_policy=CONSTANT_WEIGHT,
  l2_regularization=1e-4,
  optimiser=Adam(lr=5e-2),
  batch_size=128,
  loss_computation_batch_size=2048,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=5_000,
  num_checkpoints=1)

params = Params(
  arena=nothing,
  self_play=self_play,
  learning=learning,
  num_iters=50,
  memory_analysis=nothing,
  ternary_rewards=false,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(80_000))

benchmark_sim = SimParams(
  arena.sim;
  num_games=200,
  num_workers=8,
  batch_size=4)

benchmark = [
  Benchmark.Single(
    Benchmark.Full(self_play.mcts),
    benchmark_sim),
  Benchmark.Single(
    Benchmark.NetworkOnly(),
    benchmark_sim)]

experiment = Experiment(
  "grid-world", GameSpec(), params, Network, netparams, benchmark)