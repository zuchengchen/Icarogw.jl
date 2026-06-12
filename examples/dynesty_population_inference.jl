import Pkg
using Random

icarogw_path = normpath(joinpath(@__DIR__, ".."))
dynesty_path = normpath(joinpath(@__DIR__, "..", "..", "Dynesty.jl"))
if !isdir(dynesty_path)
    println("Skipping Dynesty example: local ../Dynesty.jl checkout not found.")
    exit()
end

try
    Pkg.activate(; temp=true)
    Pkg.develop(path=icarogw_path)
    Pkg.develop(path=dynesty_path)
    @eval using Icarogw
    @eval using Dynesty

    rng = MersenneTwister(2026)
    data = simulate_population_data(rng, SimplePowerLawPopulation(); nevents=1, nsamples=48, ndetected=64, ntotal=500, zmax=0.5)
    problem = dynesty_problem(SimplePowerLawPopulation, data; zmax=1.0)

    sampler = Dynesty.NestedSampler(
        problem.loglikelihood,
        problem.prior_transform,
        problem.ndim;
        nlive=32,
        bound=:multi,
        sample=:unif,
        rng,
    )
    Dynesty.run_nested!(sampler; maxiter=40, dlogz=nothing, print_progress=false)
    res = Dynesty.results(sampler)
    println("Dynesty logz = ", res.logz[end])
    println("samples = ", size(res.samples))
catch err
    println("Skipping Dynesty example: local ../Dynesty.jl could not be used.")
    println(err)
end
