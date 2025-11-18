"""规范相关：对易子与规范投影。"""
function ad(A::TensorMap, Z::TensorMap)
    @tensor adAZ[s, ; a, b] := A[s, k, b] * Z[k, a] - A[s, a, k] * Z[b, k]
    adAZ
end

# inner(Z,term) === ⟨Z,ad_A,Z⟩, twist is for the inner product
function ad_adj(A::TensorMap, W::TensorMap)
    @tensor term1[k; a] := conj(A[s, k, b]) * W[s, ; a, b]
    @tensor term2[b; k] := conj(A[s, a, k]) * W[s, ; a, b]
    term = term1 - term2
    term = twist(term, filter(x -> isdual(space(term, x)), allind(term)))
    return term
end

function ad_adj(rdm::TensorMap, A::TensorMap, W::TensorMap)
    @tensor term1[k; a] := conj(A[s, k, b]) * W[s, ; c, d] * rdm[a, c; b, d]
    @tensor term2[b; k] := conj(A[s, a, k]) * W[s, ; c, d] * rdm[a, c; b, d]
    term = term1 - term2
    term = twist(term, filter(x -> isdual(space(term, x)), allind(term)))
    return term
end

function proj_gauge(rdm::TensorMap, A::TensorMap, G::TensorMap)

    rhs = ad_adj(rdm, A, G)
    norm_rhs = norm(rhs)
    if norm_rhs ≤ 1.0e-12
        projected = remove_ray_component(rdm, A, G)
        return projected
    end

    Aop = Z -> ad_adj(rdm, A, ad(A, Z)) + 1.0e-6 * Z
    vec, info = KrylovKit.linsolve(Aop, rhs;
        maxiter=5,
        isposdef=true,
        ishermitian=true,
    )

    B = G - ad(A, vec)
    projected = remove_ray_component(rdm, A, B)
    return projected
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

