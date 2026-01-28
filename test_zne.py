from experiment import make_executor
import experiment
from routines import light_brute_force_search, brute_force_search, adaptive_search
from noise_model_backends import get_noise_backend
from circuits import get_experiment

NOISE_MODEL="depolarizing"                   #{depolarizing, amplitude_damping, phase_damping, readout, thermal, general_zne}
CIRCUIT="ghz"                                #{ghz, mirror_circuits, rb_circuits, rotated_rb_circuits, random_clifford_t, w_state, qpe}

# Turn on / off warnings (Better to turn it off to work with several experiments)
experiment.SILENCE_WARNINGS=True

# Create a test "batch" experiment:
zne_batch_test = {
    "noise_scaling_factors": [[1, 3, 5, 7], [1, 3, 5]],  # Noise scaling values
    "noise_scaling_method": ["global", "local_random", "local_all", "identity_scaling"],
    "extrapolation": ["linear", "richardson", "polynomial", "exponential", "poly-exp", "adaptive-exp"],
}

# Create the circuit and verifying function of the experiment
circ, verify_func, ideal_result= get_experiment(CIRCUIT)
# circ, verify_func, ideal_result= get_experiment(CIRCUIT, num_qubits=5)

# Create the backend given the noise model
backend = get_noise_backend(NOISE_MODEL)
#backend=get_noise_backend(NOISE_MODEL, prob=0.005, param_amp=0.9, excited_state_population=0, param_phase=0.9, canonical_kraus=True)

exe=make_executor(backend, verify_func, shots=4096)


# Default parameters:  circuit, executor, ideal_result, zne_batch_test
# zne_batch_test is optional, if it is not provided, there is a default search space

# BRUTE FORCE SEARCH
# brute_force_search(circ, exe, ideal_result)
# brute_force_search(circ, exe, ideal_result, zne_batch_test)


# LIGHT BRUTE FORCE SEARCH
# max_iter=10 # optional
# light_brute_force_search(circ , exe, ideal_result)
# light_brute_force_search(circ , exe, ideal_result, zne_batch_test)
# light_brute_force_search(circ , exe, ideal_result, zne_batch_test, max_iter)


# ADAPTATIVE ROUTINE (usually the user will not give the experiment parameters)
# adaptive_search(circ , exe, ideal_result, zne_batch_test)
adaptive_search(circ , exe, ideal_result)
max_factors=10 # (optional) maximum number of scale factors
tolerance=1e-4 # (optional) if the improvement is lower that the tolerance, stop expanding the list
#adaptive_search(circ , exe, ideal_result, tolerance, max_factors)





