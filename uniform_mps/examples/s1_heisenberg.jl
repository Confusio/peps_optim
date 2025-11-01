#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
try
    using MKL
catch
    @info "MKL not available; using default BLAS"
end
using UniformMPS
using TensorKit, Serialization, LinearAlgebra

# 构造随机 A :: TensorMap(P ← W_R ⊗ W_L)
A0 = TensorMap(randn, ComplexF64, P, WR ⊗ WL)
serialize("A_fs.bin", A0)

h = two_site_h(J=1.0)

for i = 1:50
    A0 = deserialize("A_fs.bin")
    A0 = regular(A0)
    A0 = A0 / norm(A0)

    e0 = energy_density(A0, A0, h)
    @info "initial energy density" e0

    A_euclid, e_euclid, info_euclid = optimize_lbfgs(A0, h; maxiter=10, g_tol=1e-8)
    serialize("A_fs.bin", A_euclid[1])
end

# @info "euclidean LBFGS energy" e_euclid
# @info "euclidean iterations" info_euclid.iterations
