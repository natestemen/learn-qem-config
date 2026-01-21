import itertools
import json

import jsonschema
import numpy as np
from mitiq import zne
from mitiq.zne.scaling.folding import fold_all, fold_gates_at_random, fold_global
from mitiq.zne.scaling.identity_insertion import insert_id_layers
from mitiq.zne.scaling.layer_scaling import get_layer_folding
from qiskit import QuantumCircuit, transpile
from noise_model_backends import get_noise_backend
from circuits import get_experiment

NOISE_MODEL="depolarizing"                   #{depolarizing, amplitude_damping, phase_damping, readout, thermal, general_zne}
CIRCUIT="mirror_circuits"                                  #{ghz, mirror_circuits, rb_circuits, rotated_rb_circuits, random_clifford_t, w_state, qpe}


# Load Schema:
def load(schema_path):
    with open(schema_path, "r") as file:
        return json.load(file)

zne_schema = load("zne.json")

# Create a test "batch" experiment:
zne_batch_test = {
    "noise_scaling_factors": [
        [1, 1.25, 1.5],
        [1, 2, 3],
        [2, 4, 6],
    ],  # Noise scaling values
    #"noise_scaling_method": ["global"],  # Folding method
    "noise_scaling_method": ["global", "local_all", "local_random"],
    "extrapolation": ["polynomial", "linear"],  # Extrapolation method
}

# Create single experiments from batch object:

# Define a function to make all combinations of experiments from a "batch dictionary" which should be formatted like the test case above
def make_experiment_list(batch_dict):
    # Initialize empty list where we will append each new experiment
    exp_list = []

    # Make list with all combinations of key values from our "batch dictionary"
    combo_list = list(
        itertools.product(
            batch_dict["noise_scaling_factors"],
            batch_dict["noise_scaling_method"],
            batch_dict["extrapolation"],
        )
    )
    # Iterate over the list
    for k in range(len(combo_list)):
        # Initialize single experiment dictionary with keys
        exp = {
            x: set()
            for x in ["noise_scaling_factors", "noise_scaling_method", "extrapolation"]
        }

        # Map each key to its unique value from our list of combinations
        exp["noise_scaling_factors"] = combo_list[k][0]
        exp["noise_scaling_method"] = combo_list[k][1]
        exp["extrapolation"] = combo_list[k][2]

        # Pull out each experiment
        exp_list.append(exp)

    # Returns a list of experiments
    return exp_list

make_experiment_list(zne_batch_test)

# Validate each experiment in our batch object:

# Create function that takes in a batch dictionary and schema to validate against
def batch_validate(batch_dict, schema):
    # Initialize empty list for validation results
    validation_results = []

    # Create list of experiments
    formatted_batch = make_experiment_list(batch_dict)

    # Iterate over list of experiments and individually validate
    for k in formatted_batch:
        try:
            jsonschema.validate(instance=k, schema=schema)
            result = "validation passed"
        except jsonschema.exceptions.ValidationError as e:
            result = "validation failed"
        except jsonschema.exceptions.SchemaError as e:
            result = "schema validation failed"
        # Pull out validation results
        validation_results.append(result)

    return validation_results

batch_validate(zne_batch_test, zne_schema)

noise_scaling_map = {
    "global": fold_global,
    "local_random": fold_gates_at_random,
    "local_all": fold_all,
    "layer": get_layer_folding,
    "identity_scaling": insert_id_layers,
}


def extrapolation_map(single_exp):
    ex_map = {
        "linear": zne.inference.LinearFactory(
            scale_factors=single_exp["noise_scaling_factors"]
        ),
        "richardson": zne.inference.RichardsonFactory(
            scale_factors=single_exp["noise_scaling_factors"]
        ),
        "polynomial": zne.inference.PolyFactory(
            scale_factors=single_exp["noise_scaling_factors"], order=2
        ),
        "exponential": zne.inference.ExpFactory(
            scale_factors=single_exp["noise_scaling_factors"]
        ),
        "poly-exp": zne.inference.PolyExpFactory(
            scale_factors=single_exp["noise_scaling_factors"], order=1
        ),
        "adaptive-exp": zne.inference.AdaExpFactory(
            scale_factor=single_exp["noise_scaling_factors"][1], steps=4, asymptote=None
        ),
    }
    return ex_map[single_exp["extrapolation"]]

def make_executor(backend, verify, shots=4096):

    def executor(circuit):
        qc = circuit.copy()
        qc.measure_all()
        transpiled_qc = transpile(qc, backend=backend, optimization_level=0)

        result = backend.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts(transpiled_qc)
        
        # Use verify function adapted to the experiment to calculate the expectation value 
        expectation_value = verify(counts)

        return expectation_value
    
    return executor

def batch_execute(batch_dict, circuit, executor):
    # Define list of experiments
    formatted_batch = make_experiment_list(batch_dict)

    # Initialize list to append expectation values into
    exp_val_list = []

    # Iterate over each experiment
    for k in formatted_batch:
        # tmp_executor = functools.partial(executor, noise_model=...)
        exp_val = zne.execute_with_zne(
            circuit=circuit,
            executor=executor,
            factory=extrapolation_map(k),
            scale_noise=noise_scaling_map[k["noise_scaling_method"]],
        )

        # Pull out expectation values
        exp_val_list.append(exp_val)

    return exp_val_list


# Create the circuit and verifying function of the experiment
circ, verify_func, ideal_result= get_experiment(CIRCUIT)

# Create the backend given the noise model
backend = get_noise_backend(NOISE_MODEL)
#backend=get_noise_backend(NOISE_MODEL, prob=0.005, param_amp=0.9, excited_state_population=0, param_phase=0.9, canonical_kraus=True)

exe=make_executor(backend, verify_func, shots=4096)


ideal_ev = ideal_result
noisy_ev=exe(circ)
print("ideal EV:", ideal_ev)
print(f"{NOISE_MODEL} EV:", noisy_ev)


exp_results = batch_execute(zne_batch_test, circ, exe)
for k in range(len(exp_results)):
    print("Experiment", k, "Mitigated Expectation Value:", exp_results[k - 1])


configs = make_experiment_list(zne_batch_test)

errors = np.abs(np.array(exp_results) - ideal_ev)
best_idx = np.argmin(errors)


print(f"BEST RESULT: Experiment #{best_idx + 1}")
print(f"Ideal Value:     {ideal_ev}")
print(f"Mitigated Value: {exp_results[best_idx]}")
print(f"Absolute Error:  {errors[best_idx]}")
print("WINNING CONFIGURATION:")
print(json.dumps(configs[best_idx], indent=4))



