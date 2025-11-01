## Toolbox of physical observables and helpers

"""两站点能量密度：局域哈密顿量 + 环境 (l, r) 收缩。"""
function energy_density(env::EnvMPS, A::TensorMap, Aconj::TensorMap, h_tm::TensorMap)
    @tensor numer_tensor = h_tm[s, t, u, v] *
                           (env.r[a, a1] * conj(Aconj[s, a, b]) * A[u, a1, b1]) *
                           (conj(Aconj[t, b, c]) * env.l[c, c1] * A[v, b1, c1])

    @tensor denom_tensor = (env.r[a, a1] * conj(Aconj[s, a, b]) * A[s, a1, b1]) *
                           (conj(Aconj[t, b, c]) * env.l[c, c1] * A[t, b1, c1])

    numer = numer_tensor / denom_tensor
    imag_part = imag(numer)
    if !isapprox(imag_part, 0; atol=1e-10)
        Zygote.ignore() do
            @warn "Energy should be real" imag_part = imag_part
        end
    end

    return real(numer)
end

function energy_density(A::TensorMap, Aconj::TensorMap, h_tm::TensorMap)
    l, r = fixedpoints(A, Aconj)
    @tensor numer_tensor = h_tm[s, t, u, v] *
                           (r[a, a1] * conj(Aconj[s, a, b]) * A[u, a1, b1]) *
                           (conj(Aconj[t, b, c]) * l[c, c1] * A[v, b1, c1])

    @tensor denom_tensor = (r[a, a1] * conj(Aconj[s, a, b]) * A[s, a1, b1]) *
                           (conj(Aconj[t, b, c]) * l[c, c1] * A[t, b1, c1])

    numer = numer_tensor / denom_tensor
    imag_part = imag(numer)
    if !isapprox(imag_part, 0; atol=1e-10)
        Zygote.ignore() do
            @warn "Energy should be real" imag_part = imag_part
        end
    end

    return real(numer)
end

