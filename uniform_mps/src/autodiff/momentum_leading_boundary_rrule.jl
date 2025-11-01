const momentum_leading_boundary_scaling = Ref((alpha = 1.0, beta = 1.0))

function set_momentum_leading_boundary_scaling!(; alpha=momentum_leading_boundary_scaling[].alpha, beta=momentum_leading_boundary_scaling[].beta)
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

    function pullback(Δenv)
        Δ_env, is_zero = to_env(Δenv)
        if is_zero
            zr = ChainRulesCore.ZeroTangent()
            return ChainRulesCore.NoTangent(), ChainRulesCore.zero_tangent(env), zr, zr
        end

        zero_l = zero(env.l)
        zero_r = zero(env.r)
        scales = momentum_leading_boundary_scaling[]

        function solve_component(comp_env::EnvMPS)
            y_env, _ = KrylovKit.linsolve(y -> begin
                Jy_raw, _, _ = vjp(y)
                Jy_env, Jy_zero = to_env(Jy_raw)
                Jy_zero && return y
                y - Jy_env
            end, comp_env)
            _, gA_raw, gAconj_raw = vjp(y_env)
            return gA_raw, gAconj_raw
        end

        gA_total = zero(A)
        gAconj_total = zero(Aconj)

        if !iszero(Δ_env.l)
            comp_env = EnvMPS(Δ_env.l, zero_r)
            gA_l, gAconj_l = solve_component(comp_env)
            gA_total += scales.alpha * to_tensor(A, gA_l)
            gAconj_total += scales.alpha * to_tensor(Aconj, gAconj_l)
        end

        if !iszero(Δ_env.r)
            comp_env = EnvMPS(zero_l, Δ_env.r)
            gA_r, gAconj_r = solve_component(comp_env)
            gA_total += scales.beta * to_tensor(A, gA_r)
            gAconj_total += scales.beta * to_tensor(Aconj, gAconj_r)
        end

        return ChainRulesCore.NoTangent(),
            ChainRulesCore.zero_tangent(env),
            gA_total,
            gAconj_total
    end

    return env_star, pullback
end
