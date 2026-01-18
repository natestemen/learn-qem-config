import networkx as nx
import numpy as np
from qiskit.quantum_info import Statevector
from mitiq.benchmarks import generate_mirror_circuit, generate_ghz_circuit, generate_rb_circuits, generate_rotated_rb_circuits

def _build_ghz(num_qubits=3, **args):

    qc = generate_ghz_circuit(n_qubits=num_qubits, return_type="qiskit")

    # Metric: percentage of |0^n> or |1^n> measurements 

    # Free-noise result: 1.0 (measuring always |0^n> or |1^n>)
    ideal_result=1.0

    # Verifying function to compute the metric
    def verify_func(counts):
        total_shots = sum(counts.values())
        if total_shots == 0: return 0.0
        s0 = "0" * num_qubits
        s1 = "1" * num_qubits
        correct = counts.get(s0, 0) + counts.get(s1, 0)
        return correct / total_shots
    
    return qc, verify_func, ideal_result 

def _build_mirror_circuits(nlayers=5, two_qubit_gate_prob=1.0, connectivity_graph=None, two_qubit_gate_name='CNOT', seed=None, return_type="qiskit",**args):
    
    # Default topology 
    topology = connectivity_graph
    if topology is None:
        n = 7
        topology = nx.complete_graph(n)
    
    qc, correct_bitstring = generate_mirror_circuit(nlayers=nlayers,two_qubit_gate_prob=two_qubit_gate_prob,connectivity_graph=topology,two_qubit_gate_name=two_qubit_gate_name,seed=seed,return_type=return_type )
    
    # Metric: percentage of intial state measurements 

    # Free-noise result: 1.0 (measuring always the initial state)
    ideal_result=1.0

    # Verifying function to compute the metric
    def verify_func(counts):
        total_shots = sum(counts.values())
        if total_shots == 0: return 0.0
        target = "".join(str(x) for x in correct_bitstring[::-1])
        return counts.get(target, 0) / total_shots

    return qc, verify_func, ideal_result

def _build_rb_circuits(n_qubits=1, num_cliffords=25, seed=None, return_type="qiskit", **args):
    
    qc_list = generate_rb_circuits(n_qubits=n_qubits, num_cliffords=num_cliffords, seed=seed, return_type=return_type)
    qc=qc_list[0]

    # Metric: percentage of |0^n> measurements 

    # Free-noise result: 1.0 (measuring always |0^n>)
    ideal_result=1.0

    # Verifying function to compute the metric
    def verify_func(counts):
        total_shots = sum(counts.values())
        if total_shots == 0: return 0.0
        target = "0" * n_qubits
        return counts.get(target, 0) / total_shots

    return qc, verify_func, ideal_result

def _build_rotated_rb_circuits(n_qubits=1, num_cliffords=25, theta=np.pi/2, seed=None, return_type="qiskit", **args):

    # Generate the list of circuits
    qc_list = generate_rotated_rb_circuits(n_qubits=n_qubits, num_cliffords=num_cliffords, theta=theta, trials=1, return_type=return_type, seed=seed)
    qc = qc_list[0]

    # Metric: Probability of measuring the ground state |0^n> 
    
    # Free-noise result: probability of measuring the ground state |0^n> in the ideal circuit 
    state = Statevector.from_instruction(qc)
    ideal_result = np.abs(state.data[0])**2

    # Verifying function to compute the metric
    def verify_func(counts):
        total_shots = sum(counts.values())
        if total_shots == 0: return 0.0 
        target = "0" * n_qubits  
        return counts.get(target, 0) / total_shots

    return qc, verify_func, ideal_result

CIRCUIT_MAP = {
    "ghz": _build_ghz,
    "mirror_circuits": _build_mirror_circuits,
    "rb_circuits": _build_rb_circuits,
    "rotated_rb_circuits": _build_rotated_rb_circuits
}

def get_experiment(name, **args):
    if name in CIRCUIT_MAP:
        return CIRCUIT_MAP[name](**args)
