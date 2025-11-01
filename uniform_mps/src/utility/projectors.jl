"""构造投影器与正则化工具。"""

function regular(As::TensorMap)
    As, S1_prev = su_iter(As)
    tol = 1.0e-12
    for _ in 1:1000
        As, S1 = su_iter(As)
        diffs = norm(S1 - S1_prev) / norm(S1)
        if diffs < tol
            break
        end
        S1_prev = S1
    end
    return As
end

function su_iter(As::TensorMap)
    @tensor L[s, WL; WR,] := As[s, ; WR, WL]
    @tensor R[WL, ; s, WR] := As[s, ; WR, WL]
    PL1, PR1, S1 = build_projectors(L, R)
    @tensor As[s, ; WR, WL] := As[s, ; WR0, WL0] * PL1[WR0; WR] * PR1[WL; WL0]
    return As / norm(As), S1
end

function build_projectors(L::AbstractTensorMap, R::AbstractTensorMap)
    L = deepcopy(L) / norm(L)
    R = deepcopy(R) / norm(R)
    if dim(codomain(L)) > dim(domain(L))
        _, L = leftorth!(L)
        R, _ = rightorth!(R)
    end
    LR = L ⊙ R
    n_factor = norm(LR)
    LR = LR / n_factor
    U, S, V = tsvd!(LR)
    norm_factor = sqrt(n_factor)
    isqS = sdiag_pow(S, -0.5)
    PL = R * V' * isqS / norm_factor
    PR = isqS * U' * L / norm_factor
    return PL, PR, S
end

⊙(t1::AbstractTensorMap, t2::AbstractTensorMap) = twist(t1, filter(i -> !isdual(space(t1, i)), domainind(t1))) * t2

function sdiag_pow(
    s::AbstractTensorMap{T,S,1,1}, pow::Real; tol::Real=eps(scalartype(s))^(3 / 4)
) where {T,S}
    # Relative tol w.r.t. largest singular value (use norm(∘, Inf) to make differentiable)
    tol *= norm(s, Inf)
    spow = similar(s)
    for (k, b) in blocks(s)
        copyto!(
            block(spow, k), LinearAlgebra.diagm(_safe_pow.(LinearAlgebra.diag(b), pow, tol))
        )
    end
    return spow
end

_safe_pow(a::Number, pow::Real, tol::Real) = (pow < 0 && abs(a) < tol) ? zero(a) : a^pow

