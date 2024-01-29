using QuantumOptics

struct Lindblad{TH, TC<:Tuple, TE<:Tuple}
    H::TH
    c_ops::TC
    e_ops::TE
end

function qihao_lindblad(;
            Omega_C = 6.20 * 2 *pi,            # Resonator Frequency (in angular frequency)
            Omega_1 = 5.50 * 2 * pi,           # Qubit Frequency (in angular frequency)
            J_C1 = 60 * 2 * pi/1000,           # JC Coupling Strength Qubit and Resonator
            Alpha_1 = -247 * 2 * pi/1000,      # Anharmonicity of Qubit
            # Chi = 0.4 * 2 * pi/1000         # Dispersive Shift (Roughly equals J_C1**2/Delta)
            Gamma_1 = 0.05 /1000,             # Gamma_1, Qubit Decoherence Rate (20.0 us)
            Gamma_phi = 0.5 /1000,            # Gamma_phi, Qubit Dephasing Rate (2.0 us)
            Amp = 0.080659, # A Low Readout Power
            Q_C,         # Trancute the Cavity Hilbert Space
            Q_1,         # Trancute the Qubit Hilbert Space
            Freq = 6.20211134 * (2*pi), # Dispersive Readout Frequency
        )          # Kappa, Resonator Decay Rate
    Delta = abs(Omega_C - Omega_1)     # Detuning bewteen Qubit and Resonator
    Kappa = 2* Omega_C/7500
    K_1 = Alpha_1 * (J_C1/Delta)^4  # Kerr Self-interaction for the Dispersive Model
    Chi_1 = 2 * J_C1^2* Alpha_1/(Delta*(Delta+Alpha_1)) # Analytical Dispersive Shift

    # Define the Operators
    a = tensor(destroy(Q_C), one(Q_1))    # Cavity Destroy Operator
    b = tensor(one(Q_C), destroy(Q_1))    # Qubit Destroy Operator

    Proj_00 = tensor(one(Q_C), basisstate(Q_1,1) |> projector)
    Proj_01 = tensor(one(Q_C), basisstate(Q_1,2) |> projector)
    Proj_02 = tensor(one(Q_C), basisstate(Q_1,3) |> projector)

    Omega_C_tilde = 1/2 * (Omega_C + Omega_1 + sqrt(Delta^2 + 4 * J_C1^2))
    Omega_1_tilde = 1/2 * (Omega_C + Omega_1 - sqrt(Delta^2 + 4 * J_C1^2))

    # The Collapse Operators
    c_ops = Any[sqrt(Kappa) * a,
            sqrt(Gamma_1) * b,
            sqrt(Gamma_phi/2)* b' * b]
    # The Expectation Operators
    e_ops = Any[a' + a, -1im * (a' - a) , a' * a, Proj_00, Proj_01, Proj_02]

    H_0 =  (Omega_C_tilde-Freq) * a' * a + (Omega_1_tilde-Freq) * b' * b +
            Alpha_1/2 * b' * b' * b * b + K_1/2 * a' * a' * a * a +
            Chi_1 * a' * a * b' *b
    H_D = Amp * (a' + a)
    H = H_0 + H_D
    return Lindblad(H, (c_ops...,), (e_ops...,))
end

function simulate(state, lin::Lindblad;
            T_tot = 2000,                      # Total Time for Readout(ns)
        )          # Kappa, Resonator Decay Rate
    T = 0:0.25:T_tot
    rho0 = dm(state)
    return timeevolution.master(T, rho0, lin.H, lin.c_ops)
end

using Plots

# Initial State
Q_C = FockBasis(59)         # Trancute the Cavity Hilbert Space
Q_1 = FockBasis(4)         # Trancute the Qubit Hilbert Space
State_00 = tensor(basisstate(Q_C,1), basisstate(Q_1,1))
State_01 = tensor(basisstate(Q_C,1), basisstate(Q_1,2))
lin = qihao_lindblad(; Q_C, Q_1)
ts1, rhos1 = simulate(State_00, lin; T_tot=2000)
ts2, rhos2 = simulate(State_01, lin; T_tot=2000)
es11 = [real(expect(lin.e_ops[1], rho)) for rho in rhos1]
es12 = [real(expect(lin.e_ops[2], rho)) for rho in rhos1]
es21 = [real(expect(lin.e_ops[1], rho)) for rho in rhos2]
es22 = [real(expect(lin.e_ops[2], rho)) for rho in rhos2]

using Plots
plot(es11, es12; label="g", xlim=(-15, 15), ylim=(0, 9))
plot!(es21, es22; label="e", xlim=(-15, 15), ylim=(0, 9))

