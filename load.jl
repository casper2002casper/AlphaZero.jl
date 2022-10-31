using CUDA, Distributed
ngpu=length(devices())
p_per_gpu = 1
addprocs((ngpu)*p_per_gpu; exeflags="--project")

@everywhere using CUDA
# assign devices
asyncmap((zip(workers(), repeat(collect(devices())[1:ngpu], p_per_gpu)))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
    end
end