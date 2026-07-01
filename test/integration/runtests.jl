using Test

const RUN_PUBLIC = get(ENV, "ICAROGW_RUN_PUBLIC_INTEGRATION", "0") in ("1", "true", "TRUE", "yes", "YES")
const CACHE_DIR = get(ENV, "ICAROGW_INTEGRATION_CACHE", joinpath(@__DIR__, "data"))

@testset "integration runner" begin
    @test isdir(@__DIR__)
    if RUN_PUBLIC
        @info "Public integration tests requested" cache=CACHE_DIR
        mkpath(CACHE_DIR)
        @test isdir(CACHE_DIR)
        @info "No public-data integration suites are registered yet; add them with the relevant migration phase."
    else
        @info "Skipping public-data integration suites; set ICAROGW_RUN_PUBLIC_INTEGRATION=1 to opt in."
        @test true
    end
end
