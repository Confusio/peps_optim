using Test
using Random
using LinearAlgebra
using TensorOperations
using TensorKit
using KrylovKit
using UniformMPS

# ---------- 基础设置 ----------
Random.seed!(0xface)

const C  = ComplexF64
const P  = ComplexSpace(3)
const WR = ComplexSpace(30)
const WL = dual(WR)

A = TensorMap(randn, C, P, WR ⊗ WL)
A = UniformMPS.regular(A)
A = A / norm(A)

env = UniformMPS.initial_env(A, A)
rdm = UniformMPS.reduced_density_matrix(env, A, A)
state = (A, env)

const hmat = UniformMPS.two_site_h(J=1.0)

energy_trial_env(Atrial) = UniformMPS.initial_env(Atrial, Atrial)
energy_trial_value(Atrial) = real(UniformMPS.energy_density(energy_trial_env(Atrial), Atrial, Atrial, hmat))

function rand_horizontal()
    raw = TensorMap(randn, C, codomain(A), domain(A))
    vec = UniformMPS.remove_ray_component(rdm, A, raw)
    for _ in 1:10
        if norm(vec) ≥ 1.0e-9
            return vec
        end
        raw = TensorMap(randn, C, codomain(A), domain(A))
        vec = UniformMPS.remove_ray_component(rdm, A, raw)
    end
    return vec
end

@testset "obj_and_grad directional derivative (euclid)" begin
    result = UniformMPS.obj_and_grad(state, hmat)
    e, grad = result
    B = rand_horizontal()
    ϵ = 1.0e-6
    fd = energy_trial_value(A + ϵ * B) - energy_trial_value(A - ϵ * B)
    fd /= (2ϵ)
    lhs = fd
    rhs = UniformMPS.euclid_inner(state, grad, B)
    denom = max(max(abs(lhs), abs(rhs)), 1.0e-12)
    @test abs(lhs - rhs) ≤ 1.0e-3 * denom
end

@testset "rdm_precondition residual" begin
    η = TensorMap(randn, C, codomain(A), domain(A))
    y = UniformMPS.rdm_precondition(state, η)

    λ = 1.0e-4
    Afun(v) = begin
        @tensor term[s, ; a, b] := rdm[a, c; b, d] * v[s, ; c, d]
        term + λ * v
    end

    residual = norm(Afun(y) - η) / max(norm(η), 1.0e-12)
    @test residual ≤ 5.0e-3

    cos = begin
        @tensor sAy := conj(A[s, a, b]) * rdm[a, c; b, d] * y[s, c, d]
        normA = sqrt(max(real(@tensor conj(A[s, a, b]) * rdm[a, c; b, d] * A[s, c, d]), 0.0))
        normY = sqrt(max(real(@tensor conj(y[s, a, b]) * rdm[a, c; b, d] * y[s, c, d]), 0.0))
        normA * normY > 0 ? abs(sAy) / (normA * normY) : 0.0
    end
    @test cos ≤ 1.0e-6
end

@testset "rdm_precondition zero input" begin
    zero_vec = zero(rand_horizontal())
    @test UniformMPS.rdm_precondition(state, zero_vec) == zero_vec
end

@testset "retract properties" begin
    η = rand_horizontal()
    α = 1.0e-3
    (state_new, _) = UniformMPS.retract(state, η, α)
    Anew, envnew = state_new
    @test abs(norm(Anew) - 1) ≤ 1.0e-10

    e0 = energy_trial_value(A)
    e1 = energy_trial_value(Anew)
    @test e1 ≤ e0 + 1.0e-6
end

@testset "optimize_lbfgs short run" begin
    e0 = energy_trial_value(A)
    (_, fx, info) = UniformMPS.optimize_lbfgs(A, hmat; maxiter=5, g_tol=1.0e-6)
    @test fx ≤ e0 + 1.0e-6
    @test info.iterations ≤ 5
end
