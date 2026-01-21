from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, phase_damping_error, amplitude_damping_error, ReadoutError, thermal_relaxation_error

def _build_depolarizing_backend(prob=0.005, **args):
    noise_model = NoiseModel()

    depolarizing_err1 = depolarizing_error(prob, num_qubits=1)
    depolarizing_err2 = depolarizing_error(prob, num_qubits=2)
    noise_model.add_all_qubit_quantum_error(depolarizing_err1, ["h", "x", "y", "z"])
    noise_model.add_all_qubit_quantum_error(depolarizing_err2, ["cx"])

    return AerSimulator(noise_model=noise_model)

def _build_amplitude_damping_backend(param_amp=0.005, excited_state_population=0, canonical_kraus=True, **args):
    noise_model = NoiseModel()

    amplitude_err = amplitude_damping_error(param_amp, excited_state_population, canonical_kraus)
    noise_model.add_all_qubit_quantum_error(amplitude_err, ["h", "x", "y", "z"])
    
    return AerSimulator(noise_model=noise_model)

def _build_phase_damping_backend(param_phase=0.005, canonical_kraus=True, **args):
    noise_model = NoiseModel()

    phase_err = phase_damping_error(param_phase, canonical_kraus)
    noise_model.add_all_qubit_quantum_error(phase_err, ["h", "x", "y", "z"])
    
    return AerSimulator(noise_model=noise_model)

def _build_readout_backend(prob=0.005, **args):
    noise_model = NoiseModel()

    readout_err = ReadoutError([[1 - prob, prob], [prob, 1 - prob]])
    noise_model.add_all_qubit_readout_error(readout_err)

    return AerSimulator(noise_model=noise_model)

def _build_thermal_relaxation_backend(T1=50e3, T2=70e3, gate_time_1q=50, gate_time_2q=150, **args):
    noise_model = NoiseModel()

    thermal_err_1q = thermal_relaxation_error(T1, T2, gate_time_1q)
    thermal_err_2q = thermal_relaxation_error(T1, T2, gate_time_2q).tensor(thermal_relaxation_error(T1, T2, gate_time_2q))
    
    noise_model.add_all_qubit_quantum_error(thermal_err_1q, ["h", "x", "y", "z"])
    noise_model.add_all_qubit_quantum_error(thermal_err_2q, ["cx"])

    return AerSimulator(noise_model=noise_model)

#This backend combines thermal and depolarizing noise for ZNE experiments
# since these are best suited for general ZNE optimization. 


def _build_general_zne_backend(prob_1q=0.002, prob_2q=0.008, T1=50e3, T2=70e3, gate_time_1q=50, gate_time_2q=300):
    noise_model = NoiseModel()
    #single qubit
    depol_1q = depolarizing_error(prob_1q, 1)
    thermal_1q = thermal_relaxation_error(T1, T2, gate_time_1q)
    err_1q = depol_1q.compose(thermal_1q)

    noise_model.add_all_qubit_quantum_error(err_1q, ["h", "x", "y", "z"])

    #two qubit
    depol_2q = depolarizing_error(prob_2q, 2)
    thermal_2q = thermal_relaxation_error(T1, T2, gate_time_2q).tensor(thermal_relaxation_error(T1, T2, gate_time_2q))
    err_2q = depol_2q.compose(thermal_2q)
    noise_model.add_all_qubit_quantum_error(err_2q, ["cx"])

    return AerSimulator(noise_model=noise_model)





NOISE_BACKEND_MAP = {
    "depolarizing": _build_depolarizing_backend,
    "amplitude_damping": _build_amplitude_damping_backend,
    "phase_damping": _build_phase_damping_backend,
    "readout": _build_readout_backend,
    "thermal": _build_thermal_relaxation_backend,
    "general_zne": _build_general_zne_backend
}

def get_noise_backend(noise_model_name, **args):
    if noise_model_name in NOISE_BACKEND_MAP:
        builder = NOISE_BACKEND_MAP[noise_model_name]
        return builder(**args)
