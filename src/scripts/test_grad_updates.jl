using Setfield
"""
    test_grad_updates(experiment; [num_games])

Run a gradient update phase using dummy data.

This is useful to ensure that the chosen hyperparameters do not
lead to _Out of Memory_ errors during training.
"""
function test_grad_updates(exp::Experiment; num_games=200, batch_size = exp.params.learning.batch_size, loss_batch_size = exp.params.learning.loss_computation_batch_size)
  exp = @set exp.params.learning.batch_size = batch_size
  exp = @set exp.params.learning.loss_computation_batch_size = loss_batch_size 
  dir="sessions/grad-updates-$(exp.name)"
  rm(dir, force=true, recursive=true)
  session = Session(exp, autosave=false, dir=dir)
  UI.Log.section(session.logger, 1, "Generating $num_games some random traces")
  for i in 1:num_games
    trace = play_game(exp.gspec, RandomPlayer())
    AlphaZero.push_trace!(session.env.memory, trace, 1.0)
  end
  UI.Log.section(session.logger, 1, "Starting gradient updates")
  AlphaZero.learning_step!(session.env, session)
end