## imports centralized in UniformMPS.jl

"""转移算符 E(X) = Σ_s A^s X (A^s)†，所有对象保持 TensorMap 形式。"""
function transfer_forward(A::TensorMap, Aconj::TensorMap, r::TensorMap)
    @tensor Y[c; d] := conj(Aconj[s, a, c]) * r[a; b] * A[s, b, d]
    Y
end

function transfer_backward(A::TensorMap, Aconj::TensorMap, l::TensorMap)
    @tensor Y[a; b] := conj(Aconj[s, a, c]) * l[c; d] * A[s, b, d]
    Y
end

"""固定点：返回 (l, r)，满足 E(r)=λ r, E†(l)=λ l。"""
function fixedpoints(A::TensorMap, Aconj::TensorMap)
    WR, WL = domain(A).spaces
    @assert Int(dim(WL)) == Int(dim(WR)) "当前实现假定左右虚腿维度一致。"

    op_r = X -> transfer_forward(A, Aconj, X)
    op_l = X -> transfer_backward(A, Aconj, X)

    T = scalartype(A)
    _, vecs_r, _ = KrylovKit.eigsolve(op_r, ones(T, WL, WL), 1, :LM)
    _, vecs_l, _ = KrylovKit.eigsolve(op_l, ones(T, WR, WR), 1, :LM)

    r = vecs_r[1]
    l = vecs_l[1]
    return l, r
end

function leading_eigenvalue(A::TensorMap, Aconj::TensorMap, l::TensorMap, r::TensorMap)
    Yr = transfer_forward(A, Aconj, r)
    @tensor λ = l[a, b] * Yr[a, b]
    return λ
end

function tmrg_iteration(env::EnvMPS, A::TensorMap, Aconj::TensorMap)
    @tensor Yr[c; d] := conj(Aconj[s, a, c]) * env.r[a; b] * A[s, b, d]
    @tensor Yl[a; b] := conj(Aconj[s, a, c]) * env.l[c; d] * A[s, b, d]
    Yl = Yl / dot(env.l, Yl)
    Yr = Yr / dot(env.r, Yr)
    return EnvMPS(Yl, Yr)
end

leading_boundary(env::EnvMPS, A::TensorMap, Aconj::TensorMap) = env

function initial_env(A::TensorMap, Aconj::TensorMap)
    l, r = fixedpoints(A, Aconj)
    return EnvMPS(l, r)
end
