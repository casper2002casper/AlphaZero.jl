
using Setfield

function learning_run(experiment::Experiment; dir=nothing, nostdout=false, start_lr=0.1, num_iters=1)
  for i in 1:num_iters
    learnrate = ConstSchedule(start_lr / (3.0^(i-1)))
    session = Session(experiment; dir, nostdout, new_params=true, learnrate)
    lrep, lperfs = Report.@timed learning_step!(session.env, session)
    @show lrep.losses
  end
  return
end
