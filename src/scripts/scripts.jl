module Scripts

  using ..AlphaZero
  using ..AlphaZero.UserInterface: Session, resume!

  include("dummy_run.jl")
  export dummy_run

  dummy_run(s::String; args...) = dummy_run(Examples.experiments[s]; args...)

  include("learning_run.jl")
  export learning_run

  learning_run(s::String; args...) = learning_run(Examples.experiments[s]; args...)

  include("just_learning.jl")
  export just_learning

  just_learning(s::String; args...) = just_learning(Examples.experiments[s]; args...)

  include("test_game.jl")
  export test_game

  test_game(e::Experiment; args...) = test_game(e.gspec; args...)

  test_game(s::String; args...) = test_game(Examples.experiments[s]; args...)

  """
      train(experiment; [dir, autosave, save_intermediate])

  Start or resume a training session.

  The optional keyword arguments are passed
  directly to the [`Session`](@ref Session(::Experiment)) constructor.
  """
  train(e::Experiment; args...) = UserInterface.resume!(Session(e; args...))

  train(s::String; args...) = train(Examples.experiments[s]; args...)

  """
      explore(experiment; [dir])

  Use the interactive explorer to visualize the current agent.
  """
  explore(e::Experiment; args...) = UserInterface.explore(Session(e; args...))

  explore(s::String; args...) = explore(Examples.experiments[s]; args...)

  function play(e::Experiment; args...)
    session = Session(e; args...)
    if GI.two_players(e.gspec)
      interactive!(session.env.gspec, AlphaZeroPlayer(session), Human())
    else
      interactive!(session.env.gspec, Human())
    end
  end

  """
      play(experiment; [dir])

  Play an interactive game against the current agent.
  """
  play(s::String; args...) = play(Examples.experiments[s]; args...)

  """
      solve(experiment; [dir])

  Solve benchmarks using trained agents
  """
  solve(s::String; args...) = solve(Examples.experiments[s]; args...)

  function solve(e::Experiment; file_location, niters = e.params.self_play.mcts.num_iters_per_turn, cpuct = e.params.self_play.mcts.cpuct, args...)
    session = Session(e; args...)
    data = open(f->read(f, String), file_location)
    state = GI.init(e.gspec, data)
    params = e.params.self_play.mcts
    params = @set params.num_iters_per_turn = niters
    params = @set params.cpuct = cpuct
    params = @set params.temperature = ConstSchedule(0.)
    play_out(e.gspec, AlphaZeroPlayer(session, mcts_params=params), state)
  end

  include("test_grad_updates.jl")

  test_grad_updates(s::String; args...) =
    test_grad_updates(Examples.experiments[s]; args...)

  
end