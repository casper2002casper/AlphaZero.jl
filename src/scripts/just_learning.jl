
using Setfield

function just_learning(experiment::Experiment; dir=nothing, nostdout=false, lr=nothing)  
    if(isnothing(lr))
        session = Session(experiment; dir, nostdout)
    else
        learnrate = ConstSchedule(lr)
        session = Session(experiment; dir, nostdout, new_params=true, learnrate)
    end
    status = nothing
    while true
        lrep, lperfs = Report.@timed learning_step!(session.env, session, init_status=status)
        if(lrep.final_status.loss.L < lrep.initial_status.loss.L)
            save(session, session.dir)
        else
            return
        end
        status = lrep.final_status
    end
end
