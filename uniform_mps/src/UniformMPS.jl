module UniformMPS

using LinearAlgebra
using TensorKit
using TensorOperations
using KrylovKit
import ChainRulesCore

# Centralized imports/aliases for methods extended in submodules/files
import Base: +, -, *, zero, similar
import LinearAlgebra: dot, mul!, rmul!, axpy!, axpby!
import VectorInterface
const VI = VectorInterface
import Zygote
import OptimKit

# 导出核心 API
export P, WL, WR,
    EnvMPS, Env,
    leading_boundary, fixedpoints,
    spin1_ops, two_site_h, energy_density,
    obj_and_grad, optimize_lbfgs, set_momentum_leading_boundary_scaling!,
    regular, GaugeProjStats


include("utility/projectors.jl")
include("environments/umps_environments.jl")
include("states/states.jl")
include("operators/localops_1d.jl")
include("algorithms/tmrg.jl")
include("toolbox.jl")
include("optim/optim_ground_state.jl")
include("autodiff/utils.jl")
include("autodiff/leading_boundary_rrule.jl")
include("autodiff/momentum_leading_boundary_rrule.jl")

end # module
