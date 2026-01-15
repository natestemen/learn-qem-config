import networkx as nx
from mitiq.benchmarks import generate_mirror_circuit, generate_ghz_circuit

def _build_ghz(num_qubits=3, **args):

    qc = generate_ghz_circuit(n_qubits=num_qubits, return_type="qiskit")
    
    # Free-noise result: |0^n> or |1^n>
    def verify_func(counts):
        total_shots = sum(counts.values())
        if total_shots == 0: return 0.0
        s0 = "0" * num_qubits
        s1 = "1" * num_qubits
        correct = counts.get(s0, 0) + counts.get(s1, 0)
        return correct / total_shots

    return qc, verify_func

def _build_mirror(nlayers=5, two_qubit_gate_prob=1.0, connectivity_graph=None, two_qubit_gate_name='CNOT', seed=None, return_type="qiskit",**args):
    
    # Default topology 
    topology = connectivity_graph
    if topology is None:
        n = 7
        topology = nx.complete_graph(n)
    
    qc, correct_bitstring = generate_mirror_circuit(nlayers=nlayers,two_qubit_gate_prob=two_qubit_gate_prob,connectivity_graph=topology,two_qubit_gate_name=two_qubit_gate_name,seed=seed,return_type=return_type )
    
    # Free-noise result: initial state given by correct_bitstring
    def verify_func(counts):
        total = sum(counts.values())
        if total == 0: return 0.0
        result = "".join(str(x) for x in correct_bitstring[::-1])
        return counts.get(result, 0) / total

    return qc, verify_func

EXPERIMENT_MAP = {
    "ghz": _build_ghz,
    "mirror_circuits": _build_mirror
}

def get_experiment(name, **args):
    if name in EXPERIMENT_MAP:
        return EXPERIMENT_MAP[name](**args)
