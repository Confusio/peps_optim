#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using UniformMPS
using TensorKit
using TensorOperations
using KrylovKit
using Zygote
using LinearAlgebra: norm
using TensorKit: inner

# Small dims to keep test fast/stable
const P3 = ComplexSpace(3)
const WR2 = ComplexSpace(10)
const WL2 = dual(WR2)

# Build a random A consistent with A[s, b, d] :: TensorMap(P3 ← WR2 ⊗ WL2)
A0 = TensorMap(randn, ComplexF64, P3, WR2 ⊗ WL2)

# Local Heisenberg two-site Hamiltonian on 3-level spin-1
Sx, Sy, Sz = UniformMPS.spin1_ops()
h_mat = kron(Sx, Sx) + kron(Sy, Sy) + kron(Sz, Sz)
h_tm = TensorMap(h_mat, P3 ⊗ P3, P3 ⊗ P3)


g1 = Zygote.gradient(x -> energy_density(A0, x, h_tm), A0)[1]

l0, r0 = fixedpoints(A0, A0)
env0 = EnvMPS(l0, r0)
f_boundary(A) = begin
    env = leading_boundary(env0, A0, A)
    energy_density(env, A0, A, h_tm)
end

g2 = Zygote.gradient(f_boundary, A0)[1]
Δg = g1 - g2
println("‖g₁‖ = ", norm(g1))
println("‖g₂‖ = ", norm(g2))
println("‖g₁‖/‖g₂‖ = ", norm(g1) / max(norm(g2), 1e-16))
println("‖Δg‖ = ", norm(Δg))
println("relative error = ", norm(Δg) / max(norm(g1), norm(g2), 1e-16))

# Finite-difference check along random direction
dir = TensorMap(randn, ComplexF64, P3, WR2 ⊗ WL2)
dir /= norm(dir)

ε = 1e-6
f = x -> energy_density(A0, x, h_tm)
fd = (f(A0 + ε * dir) - f(A0 - ε * dir)) / (2ε)
ad_dir_g2 = real(inner(dir, g2))
ad_dir_g1 = real(inner(dir, g1))
println("finite diff dir derivative = ", fd)
println("AD dir derivative (g₂) = ", ad_dir_g2)
println("difference (g₂) = ", fd - ad_dir_g2)
println("AD dir derivative (g₁) = ", ad_dir_g1)
println("difference (g₁) = ", fd - ad_dir_g1)
