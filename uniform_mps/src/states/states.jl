const C = ComplexF64
const P = ComplexSpace(3)
const WR = ComplexSpace(30)
const WL = dual(WR)

"""规范相关：对易子与规范投影。"""
function ad(A::TensorMap, Z::TensorMap)
    @tensor adAZ[s, ; a, b] := A[s, k, b] * Z[k, a] - A[s, a, k] * Z[b, k]
    adAZ
end

function ad_adj(A::TensorMap, W::TensorMap)
    @tensor term1[k; a] := conj(A[s, k, b]) * W[s, ; a, b]
    @tensor term2[b; k] := conj(A[s, a, k]) * W[s, ; a, b]
    return term1 - term2
end

function ad_adj(rdm::TensorMap, A::TensorMap, W::TensorMap)
    @tensor term1[k; a] := conj(A[s, k, b]) * W[s, ; c, d] * rdm[a, c; b, d]
    @tensor term2[b; k] := conj(A[s, a, k]) * W[s, ; c, d] * rdm[a, c; b, d]
    return term1 - term2
end

struct GaugeProjStats
    niter_eu::Int
    niter_fs::Int
    relres_eu::Float64
    relres_fs::Float64
    rKKT_eu::Float64
    rKKT_fs::Float64
    cos_ray_before::Float64
    cos_ray_after::Float64
    drop_bond::Float64
end

_info_get(info, key, default) = (hasproperty(info, key) ? getproperty(info, key) : default)

_bond_norm(rdm, W) = begin
    val = @tensor conj(W[s, ; a, b]) * rdm[a, c; b, d] * W[s, ; c, d]
    sqrt(max(real(val), 0.0))
end

function proj_gauge(rdm::TensorMap, A::TensorMap, G::TensorMap;
    λ_reg::Real=1.0e-4, tol::Real=1.0e-6, maxiter::Int=100,
    logstats::Bool=false, return_stats::Bool=false)

    rhs = ad_adj(rdm, A, G)
    norm_rhs = norm(rhs)
    if norm_rhs ≤ 1.0e-12
        projected = remove_ray_component(rdm, A, G)
        stats = GaugeProjStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return return_stats ? (projected, stats) : projected
    end

    Aop = Z -> ad_adj(rdm, A, ad(A, Z)) + λ_reg * Z
    vec, info = KrylovKit.linsolve(
        Aop,
        rhs;
        rtol=tol,
        atol=0.0,
        maxiter=maxiter,
        isposdef=true,
        ishermitian=true,
    )

    B = G - ad(A, vec)

    denom_kkt = max(norm(ad_adj(rdm, A, G)), 1.0e-12)
    rKKT = norm(ad_adj(rdm, A, B)) / denom_kkt

    normA_sq = @tensor conj(A[s, a, b]) * rdm[a, c; b, d] * A[s, c, d]
    normA = sqrt(max(real(normA_sq), 0.0))
    normB = _bond_norm(rdm, B)
    sB = @tensor conj(A[s, a, b]) * rdm[a, c; b, d] * B[s, c, d]
    cos_ray_before = normA * normB > 0 ? abs(sB) / (normA * normB) : 0.0

    projected = remove_ray_component(rdm, A, B)

    normBp = _bond_norm(rdm, projected)
    sBp = @tensor conj(A[s, a, b]) * rdm[a, c; b, d] * projected[s, c, d]
    cos_ray_after = normA * normBp > 0 ? abs(sBp) / (normA * normBp) : 0.0

    drop_bond = _bond_norm(rdm, G - projected) / max(_bond_norm(rdm, G), 1.0e-12)

    stats = GaugeProjStats(
        0,
        Int(_info_get(info, :numiter, 0)),
        NaN,
        Float64(_info_get(info, :relres, NaN)),
        0.0,
        rKKT,
        cos_ray_before,
        cos_ray_after,
        drop_bond,
    )

    if logstats
        @info "proj_gauge" stats = stats
    end

    return return_stats ? (projected, stats) : projected
end

function proj_gauge(A::TensorMap, G::TensorMap; kwargs...)
    env = initial_env(A, A)
    rdm = reduced_density_matrix(env, A, A)
    proj_gauge(rdm, A, G; kwargs...)
end

function reduced_density_matrix(env::EnvMPS, A::TensorMap, Aconj::TensorMap)
    @tensor rdm[a, c; b, d] := env.r[a; c] * env.l[b; d]
    @tensor λ = rdm[a, c; b, d] * conj(Aconj[s, a, b]) * A[s, c, d]
    return rdm / λ
end

remove_ray_component(rdm::TensorMap, A::TensorMap, B::TensorMap) = begin
    @tensor s = conj(A[s, a, b]) * rdm[a, c; b, d] * B[s, c, d]
    B - s * A
end

function norm_FS(rdm::TensorMap, A::TensorMap)
    @tensor λ = rdm[a, c; b, d] * conj(A[s, ; a, b]) * A[s, ; c, d]
    return sqrt(real(λ))
end
