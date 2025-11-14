function ChainRulesCore.rrule(::typeof(leading_boundary), env::EnvMPS, A::TensorMap, Aconj::TensorMap)
    env_star = env
    zero_env() = EnvMPS(zero(env.l), zero(env.r))
    _, vjp = Zygote.pullback(tmrg_iteration, env_star, A, Aconj)

    function to_env(Δ)
        Δu = ChainRulesCore.unthunk(Δ)
        if Δu isa ChainRulesCore.AbstractZero || Δu isa ChainRulesCore.ZeroTangent
            return zero_env(), true
        elseif Δu isa EnvMPS
            return Δu, false
        elseif Δu isa ChainRulesCore.Tangent
            lval = hasproperty(Δu, :l) ? ChainRulesCore.unthunk(getproperty(Δu, :l)) : zero(env.l)
            rval = hasproperty(Δu, :r) ? ChainRulesCore.unthunk(getproperty(Δu, :r)) : zero(env.r)
            return EnvMPS(lval, rval), false
        elseif Δu isa NamedTuple
            lval = haskey(Δu, :l) ? ChainRulesCore.unthunk(Δu.l) : zero(env.l)
            rval = haskey(Δu, :r) ? ChainRulesCore.unthunk(Δu.r) : zero(env.r)
            return EnvMPS(lval, rval), false
        elseif Δu isa Tuple
            lval = length(Δu) >= 1 ? ChainRulesCore.unthunk(Δu[1]) : zero(env.l)
            rval = length(Δu) >= 2 ? ChainRulesCore.unthunk(Δu[2]) : zero(env.r)
            return EnvMPS(lval, rval), false
        end
        throw(ArgumentError("Expected EnvMPS-compatible tangent for leading_boundary pullback, got $(typeof(Δu))"))
    end

    function pullback(Δenv)
        Δ_env, is_zero = to_env(Δenv)
        if is_zero
            zr = ChainRulesCore.ZeroTangent()
            return ChainRulesCore.NoTangent(), ChainRulesCore.zero_tangent(env), zr, zr
        end

        lin_op = function (y_env::EnvMPS)
            Jy_raw, _, _ = vjp(y_env)
            Jy_env, Jy_zero = to_env(Jy_raw)
            Jy_zero && return y_env
            y_env - Jy_env
        end

        y_env, _ = KrylovKit.linsolve(lin_op, Δ_env; rtol=1.0e-12, atol=0.0)
        _, gA, gAconj = vjp(y_env)
        return ChainRulesCore.NoTangent(),
        ChainRulesCore.zero_tangent(env),
        gA,
        gAconj
    end

    return env_star, pullback
end
