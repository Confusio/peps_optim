"""Environment container for 1D uniform MPS transfer fixed points (keeps only l, r)."""
struct EnvMPS{L,R}
    l::L
    r::R
end

# Simple rrule so AD can propagate through constructor usage (e.g. tmrg_iteration)
function ChainRulesCore.rrule(::Type{EnvMPS}, l, r)
    env = EnvMPS(l, r)
    function env_pullback(ē)
        ēu = ChainRulesCore.unthunk(ē)
        if ēu isa ChainRulesCore.AbstractZero || ēu isa ChainRulesCore.ZeroTangent
            return ChainRulesCore.NoTangent(),
                   ChainRulesCore.zero_tangent(l),
                   ChainRulesCore.zero_tangent(r)
        elseif ēu isa EnvMPS
            return ChainRulesCore.NoTangent(), ēu.l, ēu.r
        end
        throw(ArgumentError("Expected EnvMPS tangent, got $(typeof(ēu))"))
    end
    return env, env_pullback
end

function ChainRulesCore.rrule(::typeof(getproperty), env::EnvMPS, name::Symbol)
    value = getproperty(env, name)
    zero_env_l = zero(env.l)
    zero_env_r = zero(env.r)

    if name === :l
        function pullback_l(Δ̄)
            Δ = ChainRulesCore.unthunk(Δ̄)
            grad_env = EnvMPS(Δ, zero_env_r)
            return ChainRulesCore.NoTangent(), grad_env, ChainRulesCore.NoTangent()
        end
        return value, pullback_l
    elseif name === :r
        function pullback_r(Δ̄)
            Δ = ChainRulesCore.unthunk(Δ̄)
            grad_env = EnvMPS(zero_env_l, Δ)
            return ChainRulesCore.NoTangent(), grad_env, ChainRulesCore.NoTangent()
        end
        return value, pullback_r
    end

    throw(ArgumentError("No rrule implemented for getproperty(::EnvMPS, $(name))"))
end

function ChainRulesCore.rrule(::typeof(getproperty), env::EnvMPS, v::Val)
    return ChainRulesCore.rrule(getproperty, env, v.val)
end

# Backward-compatible aliases
const Env = EnvMPS
const MPSEnv = EnvMPS

# Vector-like ops and interfaces (moved from utility/envmps_vector.jl)

*(α::Number, e::EnvMPS) = EnvMPS(α * e.l, α * e.r)
*(e::EnvMPS, α::Number) = α * e
+(a::EnvMPS, b::EnvMPS) = EnvMPS(a.l + b.l, a.r + b.r)
-(a::EnvMPS, b::EnvMPS) = EnvMPS(a.l - b.l, a.r - b.r)
zero(e::EnvMPS) = EnvMPS(zero(e.l), zero(e.r))

function dot(a::EnvMPS, b::EnvMPS)
    s = TensorKit.inner(a.l, b.l) + TensorKit.inner(a.r, b.r)
    return real(s)
end


# In-place linear algebra style operations to support iterative solvers
function mul!(edst::EnvMPS, esrc::EnvMPS, α::Number)
    edst.l .= α .* esrc.l
    edst.r .= α .* esrc.r
    return edst
end

function rmul!(e::EnvMPS, α::Number)
    e.l .*= α
    e.r .*= α
    return e
end

function axpy!(α::Number, e₁::EnvMPS, e₂::EnvMPS)
    e₂.l .+= α .* e₁.l
    e₂.r .+= α .* e₁.r
    return e₂
end

function axpby!(α::Number, e₁::EnvMPS, β::Number, e₂::EnvMPS)
    e₂.l .= α .* e₁.l .+ β .* e₂.l
    e₂.r .= α .* e₁.r .+ β .* e₂.r
    return e₂
end

# Optional: make a similar container; keep scalar initialized to zero
Base.similar(e::EnvMPS) = EnvMPS(similar(e.l), similar(e.r))

# VectorInterface compatibility (useful for KrylovKit internals)

function VI.scalartype(::Type{EnvMPS{L,R}}) where {L,R}
    S₁ = VI.scalartype(L)
    S₂ = VI.scalartype(R)
    return promote_type(S₁, S₂)
end

function VI.zerovector(env::EnvMPS, ::Type{S}) where {S<:Number}
    EnvMPS(zero(env.l), zero(env.r))
end
function VI.zerovector!(env::EnvMPS)
    EnvMPS(zero(env.l), zero(env.r))
end
VI.zerovector!!(env::EnvMPS) = VI.zerovector!(env)

function VI.scale(env::EnvMPS, α::Number)
    EnvMPS(VI.scale(env.l, α), VI.scale(env.r, α))
end
function VI.scale!(env::EnvMPS, α::Number)
    EnvMPS(VI.scale(env.l, α), VI.scale(env.r, α))
end
function VI.scale!(env₁::EnvMPS, env₂::EnvMPS, α::Number)
    EnvMPS(VI.scale(env₂.l, α), VI.scale(env₂.r, α))
end
VI.scale!!(env::EnvMPS, α::Number) = VI.scale!(env, α)
VI.scale!!(env₁::EnvMPS, env₂::EnvMPS, α::Number) = VI.scale!(env₁, env₂, α)

function VI.add(env₁::EnvMPS, env₂::EnvMPS, α::Number, β::Number)
    EnvMPS(VI.add(env₁.l, env₂.l, α, β), VI.add(env₁.r, env₂.r, α, β))
end
function VI.add!(env₁::EnvMPS, env₂::EnvMPS, α::Number, β::Number)
    EnvMPS(VI.add(env₁.l, env₂.l, α, β), VI.add(env₁.r, env₂.r, α, β))
end
VI.add!!(env₁::EnvMPS, env₂::EnvMPS, α::Number, β::Number) = VI.add!(env₁, env₂, α, β)

function VI.inner(env₁::EnvMPS, env₂::EnvMPS)
    # Tensor parts via TensorKit.inner
    return real(TensorKit.inner(env₁.l, env₂.l) + TensorKit.inner(env₁.r, env₂.r))
end
VI.norm(env::EnvMPS) = sqrt(VI.inner(env, env))
# (No LinearAlgebra.norm method; KrylovKit uses VectorInterface.norm)
