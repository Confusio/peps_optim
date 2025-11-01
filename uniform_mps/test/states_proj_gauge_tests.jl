using Test
using Random
using LinearAlgebra
using TensorKit
using TensorOperations
using KrylovKit
using UniformMPS

Random.seed!(0xface)

A = TensorMap(randn, ComplexF64, P, WR ⊗ WL)
A = UniformMPS.regular(A)
A = A / LinearAlgebra.norm(A)
env = UniformMPS.initial_env(A, A)
rdm = UniformMPS.reduced_density_matrix(env, A, A)

bond_dot(rdm, W1, W2) = begin
    val = @tensor conj(W1[s, ; a, b]) * rdm[a, c; b, d] * W2[s, ; c, d]
    real(val)
end

bond_norm(rdm, W) = sqrt(max(bond_dot(rdm, W, W), 0.0))

euclid_dot(W1, W2) = real(TensorKit.inner(W1, W2))

const λ_reg = 1.0e-6

@testset "proj_gauge bond-space properties" begin
    # 1. Adjoint consistency
    @testset "adjoint consistency" begin
        for _ in 1:5
            Z = TensorMap(randn, ComplexF64, WR, WR)
            W = TensorMap(randn, ComplexF64, codomain(A), domain(A))
            lhs = bond_dot(rdm, UniformMPS.ad(A, Z), W)
            rhs = real(TensorKit.inner(Z, UniformMPS.ad_adj(rdm, A, W)))
            @test isapprox(lhs, rhs; atol=5e-11, rtol=5e-11)
        end
    end

    # 2. KKT / horizontality
    @testset "horizontality" begin
        B = TensorMap(randn, ComplexF64, codomain(A), domain(A))
        B_hor = UniformMPS.proj_gauge(rdm, A, B)
        baseline = max(LinearAlgebra.norm(UniformMPS.ad_adj(rdm, A, B)), 1.0e-12)
        residual = LinearAlgebra.norm(UniformMPS.ad_adj(rdm, A, B_hor))
        ratio = residual / baseline
        @test ratio ≤ 1.0e-2
    end

    # 3. Pure gauge annihilation
    @testset "pure gauge" begin
        Z = TensorMap(randn, ComplexF64, WR, WR)
        B = UniformMPS.ad(A, Z)
        B_proj = UniformMPS.proj_gauge(rdm, A, B)
        ratio = bond_norm(rdm, B_proj) / max(bond_norm(rdm, B), 1.0e-12)
        @test ratio ≤ 5.0e-2
    end

    # 4. Idempotence
    @testset "idempotence" begin
        B = TensorMap(randn, ComplexF64, codomain(A), domain(A))
        B_proj = UniformMPS.proj_gauge(rdm, A, B)
        B_proj2 = UniformMPS.proj_gauge(rdm, A, B_proj)
        diff = bond_norm(rdm, B_proj2 - B_proj)
        denom = max(bond_norm(rdm, B_proj), 1.0e-12)
        @test diff / denom ≤ 5.0e-2
    end

    # 5. True linear-system residual
    @testset "linear solve residual" begin
        B = TensorMap(randn, ComplexF64, codomain(A), domain(A))
        rhs = UniformMPS.ad_adj(rdm, A, B)
        Aop = Z -> UniformMPS.ad_adj(rdm, A, UniformMPS.ad(A, Z)) + λ_reg * Z
        Z_sol, info = KrylovKit.linsolve(Aop, rhs;
            rtol=1.0e-6,
            atol=0.0,
            maxiter=40,
            isposdef=true,
        )
        residual = LinearAlgebra.norm(Aop(Z_sol) - rhs)
        rhs_norm = max(LinearAlgebra.norm(rhs), 1.0e-12)
        @test residual / rhs_norm ≤ 2.0e-2
    end

    # 6. Linearity and gauge invariance
    @testset "linearity and gauge" begin
        B1 = TensorMap(randn, ComplexF64, codomain(A), domain(A))
        B2 = TensorMap(randn, ComplexF64, codomain(A), domain(A))
        α = 1.7 + 0.3im

        P1 = UniformMPS.proj_gauge(rdm, A, B1)
        P2 = UniformMPS.proj_gauge(rdm, A, B2)
        Psum = UniformMPS.proj_gauge(rdm, A, B1 + B2)
        lin_err = bond_norm(rdm, (P1 + P2) - Psum) / max(bond_norm(rdm, Psum), 1.0e-12)
        @test lin_err ≤ 5.0e-2

        Zg = TensorMap(randn, ComplexF64, WR, WR)
        Pgauge = UniformMPS.proj_gauge(rdm, A, B1 + UniformMPS.ad(A, Zg))
        Pgauge_err = bond_norm(rdm, Pgauge - P1) / max(bond_norm(rdm, P1), 1.0e-12)
        @test Pgauge_err ≤ 5.0e-2

        Pscaled = UniformMPS.proj_gauge(rdm, A, α * B1)
        scale_err = bond_norm(rdm, Pscaled - α * P1) / max(bond_norm(rdm, Pscaled), 1.0e-12)
        @test scale_err ≤ 5.0e-2
    end

    @testset "self-adjointness (bond)" begin
        B1 = TensorMap(randn, ComplexF64, codomain(A), domain(A))
        B2 = TensorMap(randn, ComplexF64, codomain(A), domain(A))
        P1 = UniformMPS.proj_gauge(rdm, A, B1)
        P2 = UniformMPS.proj_gauge(rdm, A, B2)
        lhs = bond_dot(rdm, P1, B2)
        rhs = bond_dot(rdm, B1, P2)
        @test isapprox(lhs, rhs; rtol=1e-3, atol=1e-10)
    end

    @testset "ray overlap reduction" begin
        B = TensorMap(randn, ComplexF64, codomain(A), domain(A))
        B_proj = UniformMPS.proj_gauge(rdm, A, B)
        sBproj = @tensor conj(A[s, a, b]) * rdm[a, c; b, d] * B_proj[s, c, d]
        normA = sqrt(max(real(@tensor conj(A[s, a, b]) * rdm[a, c; b, d] * A[s, c, d]), 0.0))
        normBproj = bond_norm(rdm, B_proj)
        denom = max(normA * normBproj, 1.0e-16)
        cosθ = abs(sBproj) / denom
        @test cosθ ≤ 1.0e-6
    end

    @testset "proj_gauge stats" begin
        B = TensorMap(randn, ComplexF64, codomain(A), domain(A))
        B_proj, stats = UniformMPS.proj_gauge(rdm, A, B; return_stats=true, logstats=false)
        @test isa(stats, UniformMPS.GaugeProjStats)
        @test bond_norm(rdm, B_proj - UniformMPS.proj_gauge(rdm, A, B)) ≤ 1.0e-10
        @test 0.0 ≤ stats.drop_bond ≤ 1.0 + 1e-6
        @test stats.rKKT_fs ≤ 5.0e-2
    end
end

@testset "ad_adj Euclidean properties" begin
    @testset "adjoint consistency (euclid)" begin
        for _ in 1:5
            Z = TensorMap(randn, ComplexF64, WR, WR)
            W = TensorMap(randn, ComplexF64, codomain(A), domain(A))
            lhs = euclid_dot(UniformMPS.ad(A, Z), W)
            rhs = euclid_dot(Z, UniformMPS.ad_adj(A, W))
            @test isapprox(lhs, rhs; atol=5e-11, rtol=5e-11)
        end
    end

    @testset "linear solve residual (euclid)" begin
        B = TensorMap(randn, ComplexF64, codomain(A), domain(A))
        rhs = UniformMPS.ad_adj(A, B)
        Aop = Z -> UniformMPS.ad_adj(A, UniformMPS.ad(A, Z)) + λ_reg * Z
        Z_sol, info = KrylovKit.linsolve(Aop, rhs;
            rtol=1.0e-6,
            atol=0.0,
            maxiter=40,
            isposdef=true,
        )
        residual = LinearAlgebra.norm(Aop(Z_sol) - rhs)
        rhs_norm = max(LinearAlgebra.norm(rhs), 1.0e-12)
        @test residual / rhs_norm ≤ 2.0e-2
    end
end
