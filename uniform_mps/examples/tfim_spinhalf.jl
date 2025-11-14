#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
try
    using MKL
catch
    @info "MKL not available; using default BLAS"
end
using UniformMPS
using TensorKit, Serialization, LinearAlgebra

"""Pauli matrices (σx, σy, σz) on the spin-1/2 physical space."""
function pauli_ops()
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]
    return σx, σy, σz
end

"""Two-site transverse-field Ising Hamiltonian for spin-1/2."""
function tfim_two_site(; J::Real=1.0, Γ::Real=1.0)
    σx, σy, σz = pauli_ops()
    I2 = Matrix{ComplexF64}(I, 2, 2)
    h_mat = kron(σz, σz) - (kron(σx, σx) + kron(σy, σy))
    P = ComplexSpace(2)
    TensorMap(h_mat, P ⊗ P, P ⊗ P) * J / 4.0
end

function j_model(J::Float64=1.0)
    #tj model after Marshall sign transformation
    Pspace = Vect[FermionParity](1 => 2)

    hJ = zeros(ComplexF64, Pspace ⊗ Pspace ← Pspace ⊗ Pspace)
    I = sectortype(hJ)
    # No sign frustration after Marshall sign transformation
    hJ[(I(1), I(1), dual(I(1)), dual(I(1)))][1, 2, 1, 2] = -0.5 #-0.5|↑⟩|↓⟩←|↑⟩|↓⟩
    hJ[(I(1), I(1), dual(I(1)), dual(I(1)))][2, 1, 1, 2] = 0.5 #-0.5|↓⟩|↑⟩←|↑⟩|↓⟩
    hJ[(I(1), I(1), dual(I(1)), dual(I(1)))][2, 1, 2, 1] = -0.5 #-0.5|↓⟩|↑⟩←|↓⟩|↑⟩
    hJ[(I(1), I(1), dual(I(1)), dual(I(1)))][1, 2, 2, 1] = 0.5 #-0.5|↑⟩|↓⟩←|↓⟩|↑⟩

    return hJ * J
end

# Configure global spaces for spin-1/2 TFIM
spaces = UniformMPS.set_space_config!(phys_dim=2, bond_dim=30)

Pspace = Vect[FermionParity](1 => 2)
WL = Vect[FermionParity](0 => 15, 1 => 15)
# Random initial tensor A :: TensorMap(P ← W_R ⊗ W_L)
A_path = joinpath(@__DIR__, "A_jmodel_spinhalf.bin")
A0 = TensorMap(randn, ComplexF64, Pspace, WL' ⊗ WL)
serialize(A_path, A0)

h = j_model(1.0)

for i = 1:50
    A0 = deserialize(A_path)
    A0 = regular(A0)
    A0 = A0 / norm(A0)

    e0 = energy_density(A0, A0, h)
    @info "initial TFIM energy" iteration = i energy = e0

    A_euclid, e_euclid, info_euclid = optimize_lbfgs(A0, h; maxiter=30, g_tol=1e-8)
    serialize(A_path, A_euclid[1])
end

# @info "tfim LBFGS energy" e_euclid
# @info "tfim iterations" info_euclid.iterations
