const momentum_leading_boundary_scaling = Ref((alpha = 1.0 + 0im, beta = 1.0 + 0im))

function set_momentum_leading_boundary_scaling!(; alpha=momentum_leading_boundary_scaling[].alpha, beta=conj(alpha))
    momentum_leading_boundary_scaling[] = (alpha=alpha, beta=beta)
    return nothing
end

leading_boundary(::Val{:momentum}, env::EnvMPS, A::TensorMap, Aconj::TensorMap) =
    leading_boundary(env, A, Aconj)

function ChainRulesCore.rrule(::typeof(leading_boundary), ::Val{:momentum}, env::EnvMPS, A::TensorMap, Aconj::TensorMap)
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
        throw(ArgumentError("Expected EnvMPS-compatible tangent for momentum leading_boundary pullback, got $(typeof(Δu))"))
    end

    function to_tensor(template::TensorMap, Δ)
        Δu = ChainRulesCore.unthunk(Δ)
        if Δu isa ChainRulesCore.AbstractZero || Δu isa ChainRulesCore.ZeroTangent
            return zero(template)
        end
        return Δu
    end

    apply_alpha_env(e::EnvMPS, αL, αR) = EnvMPS(αL * e.l, αR * e.r)

    function momentum_lin_op(αL::Complex, αR::Complex)
        y -> begin
            Jy_raw, _, _ = vjp(y)
            Jy_env, Jy_zero = to_env(Jy_raw)
            Jy_zero && return y
            scaled = apply_alpha_env(Jy_env, αL, αR)
            EnvMPS(y.l - scaled.l, y.r - scaled.r)
        end
    end

    function pullback(Δenv)
        Δ_env, is_zero = to_env(Δenv)
        if is_zero
            zr = ChainRulesCore.ZeroTangent()
            return ChainRulesCore.NoTangent(), ChainRulesCore.zero_tangent(env), zr, zr
        end

        αR = momentum_leading_boundary_scaling[].alpha
        αL = momentum_leading_boundary_scaling[].beta

        lin_op = momentum_lin_op(αL, αR)
        y_env, _ = KrylovKit.linsolve(lin_op, Δ_env)

        scaled_env = apply_alpha_env(y_env, αL, αR)
        env_grad_raw, gA_raw, gAconj_raw = vjp(scaled_env)

        gA = to_tensor(A, gA_raw)
        gAconj = to_tensor(Aconj, gAconj_raw)

        return ChainRulesCore.NoTangent(),
            ChainRulesCore.zero_tangent(env),
            gA,
            gAconj
    end

    return env_star, pullback
end
