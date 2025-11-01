#!/usr/bin/env julia
using UniformMPS
using TensorKit
using KrylovKit
using Zygote

# Small test spaces to keep everything stable and fast
const P2  = ComplexSpace(2)
const WR2 = ComplexSpace(3)
const WL2 = dual(WR2)

# Random test tensor (seed-free here; can be adjusted)
A0 = TensorMap(randn, ComplexF64, P2, WR2 ⊗ WL2)

f_fixed(A) = begin
    l, r = fixedpoints(A, A; tol=1e-12, maxiter=500)
    λ = UniformMPS.leading_eigenvalue(A, A, l, r)
    return real(λ)
end

f_krylov(A) = begin
    op_r = X -> UniformMPS.transfer_forward(A, A, X)
    # start vector in the correct space
    v0 = ones(ComplexF64, WL2, WL2)
    vals, _, _ = KrylovKit.eigsolve(op_r, v0, 1, :LM; tol=1e-12, maxiter=500, ishermitian=false)
    return real(vals[1])
end

ϵ1, back1 = Zygote.withgradient(f_fixed, A0)
g1 = only(back1)

ϵ2, back2 = Zygote.withgradient(f_krylov, A0)
g2 = only(back2)

diff = norm(g1 - g2)
rel = diff / max(norm(g1), norm(g2), 1e-16)
println("fixedpoints AD: ", ϵ1)
println("krylovkit   AD: ", ϵ2)
println("‖Δg‖ = ", diff, ",  rel = ", rel)
