using CUDA, Distributed
ngpu=length(devices())
addprocs(ngpu-1; exeflags="--project")

@everywhere using CUDA

# assign devices
asyncmap((zip(workers(), collect(devices())[2:ngpu]))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
    end
end