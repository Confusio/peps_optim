Zygote.@nograd KrylovKit.apply_scalartype
Zygote.@nograd KrylovKit.genapply_scalartype

function obj_and_grad(state, hmat::TensorMap)
    A, env = state
    energy(A_trial) = begin
        env_trial = leading_boundary(env, A, A_trial)
        energy_density(env_trial, A, A_trial, hmat)
    end
    e, back = Zygote.withgradient(energy, A)
    grad_euclid = 2 * only(back)

    return e, grad_euclid
end

"最简 OptimKit 接口（欧氏内积）：LBFGS 起步"
function optimize_lbfgs(A0::TensorMap, hmat::TensorMap;
    maxiter::Int=200, g_tol::Real=1e-8)
    env0 = initial_env(A0, A0)
    fg = a -> obj_and_grad(a, hmat)

    ls = OptimKit.HagerZhangLineSearch(
        maxiter=6,
        maxfg=10,
    )

    method = OptimKit.LBFGS(20; maxiter=maxiter,
        gradtol=g_tol,
        verbosity=3,
        linesearch=ls)
    x_opt, fx, grad, numfg, history = OptimKit.optimize(fg, (A0, env0), method;
        inner=euclid_inner,
        retract=retract,
        precondition=rdm_precondition
    )
    info = (; gradient=grad,
        evaluations=numfg,
        history,
        iterations=size(history, 1) - 1)
    return x_opt, fx, info
end

function real_inner(state, η₁, η₂)
    inner_vec = metric_FS(state, η₂)
    return real(dot(η₁, inner_vec))
end

function retract(state, η, α)
    A, env = state
    A_new = A + α * η
    A_new /= norm(A_new)
    env_new = initial_env(A_new, A_new)
    return (A_new, env_new), η
end

function metric_FS(state, C::TensorMap)
    A, env = state
    rdm = reduced_density_matrix(env, A, A)
    @tensor term1[s, ; a, b] := rdm[a, c; b, d] * C[s, ; c, d]
    return term1
end

euclid_inner(state, η₁, η₂) = real(dot(η₁, η₂))
function rdm_precondition(state, η)
    if norm(η) ≤ 1e-14
        return η
    end
    A, env = state
    rdm = reduced_density_matrix(env, A, A)
    λ = 1.0e-6
    μ = 1.0e-2
    Afun = v -> begin
        @tensor term1[s, ; a, b] := rdm[a, c; b, d] * v[s, ; c, d]

        # term1 + μ * ad(A, ad_adj(A, v)) + λ * v
        term1 + λ * v
    end

    y, info = KrylovKit.linsolve(Afun, η;
        maxiter=5,
        # ishermitian=true,
        # isposdef=true,
    )
    # if norm(y) < 0.1
    #     y = proj_gauge(rdm, A, y)
    # end
    return y
end
