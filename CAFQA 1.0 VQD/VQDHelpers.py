from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler.passes import RemoveBarriers
from qiskit import transpile, execute
from qiskit_aer import AerSimulator, Aer
from timeit import default_timer as timer
import numpy as np
import csv
from circuit_manipulation import *
def get_energies(coeffs, paulis):
    """
    Get the energy values of a Hamiltonian experessed as linear combination of paulis
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    paulis (Iterable[String]): Corresponding Pauli strings in Hamiltonian (same order as coeffs).
    
    Returns:
    (array) the eigenvalues in ascending order
    """
    H=SparsePauliOp(data=paulis, coeffs=coeffs)
    return np.linalg.eigvalsh(H.to_matrix())

def efficientsu2(n_qubits, repetitions,initial=None):
    """
    Get the VQE ansatz with a specified number of qubits, repetition count, and initialization
    n_qubits (Int): Number of qubits in circuit.
    repetitions (Int): # ansatz repetitions.
    initial (String): Bitstring to initialize to, e.g. "01101" -> |01101> (in Qiskit ordering |10110>)
    
    Returns:
    (QuantumCircuit) the obtained circuit
    """
    qc=QuantumCircuit(n_qubits)
    if initial is not None:
        for i in range(len(initial)):
            if initial[i] == "1":
                qc.x(i)
    ansatz = EfficientSU2(num_qubits=n_qubits, entanglement='reverse_linear', reps=repetitions)
    ansatz = ansatz.decompose()
    qc=qc.compose(ansatz)
    return qc


def vqe_measurement(pauli,ansatz,parameters):
    """
    Attach the needed measurements to a provided circuit
    pauli (String): The pauli string to be measured
    ansatz (QuantumCircuit): The ansatz that's going to be used
    paraemters (Iterable[Float]): The ansatz parameters
    
    Returns:
    (QuantumCircuit): The circuit with necessary measurment added 
    """
    q=ansatz.copy()
    q=q.assign_parameters(parameters)
    n_qubits = ansatz.num_qubits
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qr, cr)
    qc.compose(q,qubits=range(0,n_qubits),inplace=True)
    #add the measurement operations
    for i, el in enumerate(pauli):
        if el == 'I':
            #no measurement for identity
            continue
        elif el == 'Z':
            qc.measure(qr[i], cr[i])
        elif el == 'X':
            qc.u(np.pi/2, 0, np.pi, qr[i])
            qc.measure(qr[i], cr[i])
        elif el == 'Y':
            qc.u(np.pi/2, 0, np.pi/2, qr[i])
            qc.measure(qr[i], cr[i])
    return qc

def all_transpiled_vqe_circuits(paulis, backend, ansatz,parameters,seed_transpiler=13):
    """
    Get a list of all circuits transpiled for a given backend
    paulis (Iterable[String]): The pauli strings to be measured
    backend (IBM backend): The backend used to simulate the circuits
    ansatz (QuantumCircuit): The ansatz used
    parameters (Iterable[Float]): The parameters for the ansatz
    seed_transpiler (Int): The seed for the transpiler. Default is 13

    Returns:
    List[QuantumCircuit] of all transpiled circuits
    """
    backend_qubits = backend.num_qubits
    n_qubits = ansatz.num_qubits
    circuit = vqe_measurement(n_qubits*'Z', ansatz,parameters)
    # transpile one circuit
    t_circuit = transpile(circuit, backend, optimization_level=3, seed_transpiler=seed_transpiler)
    # get the mapping from virtual to physical
    virtual_to_physical_mapping = {}
    for inst in t_circuit:
        if inst[0].name == 'measure':
            virtual_to_physical_mapping[t_circuit.find_bit(inst[2][0]).index] = t_circuit.find_bit(inst[1][0]).index
    # remove final measurements
    t_circuit.remove_final_measurements()
    # create all transpiled circuits
    all_transpiled_circuits = []
    for pauli in paulis:
        new_circ = QuantumCircuit(backend_qubits, n_qubits)
        new_circ.compose(t_circuit, inplace = True)
        for idx, el in enumerate(pauli):
            if el == 'I':
                continue
            elif el == 'Z':
                new_circ.measure(virtual_to_physical_mapping[idx], idx)
            elif el == 'X':
                new_circ.rz(np.pi/2, virtual_to_physical_mapping[idx])
                new_circ.sx(virtual_to_physical_mapping[idx])
                new_circ.rz(np.pi/2, virtual_to_physical_mapping[idx])
                new_circ.measure(virtual_to_physical_mapping[idx], idx)
            elif el == 'Y':
                new_circ.sx(virtual_to_physical_mapping[idx])
                new_circ.rz(np.pi/2, virtual_to_physical_mapping[idx])
                new_circ.measure(virtual_to_physical_mapping[idx], idx)
        all_transpiled_circuits.append(new_circ)
    return all_transpiled_circuits


def compute_expectations(paulis, shots, backend, noisy, ansatz, parameters,seed_transpiler=13):
    """
    Compute the expection values of the Pauli strings.
    paulis (Iterable[String]): Corresponding Pauli strings in Hamiltonian (same order as coeffs).
    shots (Int): Number of VQE circuit execution shots.
    backend (IBM backend): The backend used to simulate the circuits
    noisy (Bool): True for noisy sim and False for ideal sim
    ansatz (QuantumCircuit): The ansatz used
    parameters (Iterable[Float]): VQE parameters.
    seed_transpiler (Int): The seed for the transpiler. Default is 13

    Returns:
    List[Float] of expection value for each Pauli string.
    """
    #evaluate the circuits
    if(not noisy):
        #get all the vqe circuits
        circuits = [vqe_measurement(pauli,ansatz, parameters) for pauli in paulis]
        result = execute(circuits, backend=Aer.get_backend("qasm_simulator"), shots=shots).result()
    else:
        sim_device = AerSimulator.from_backend(backend)
        tcircs = all_transpiled_vqe_circuits(paulis, backend, ansatz,parameters,seed_transpiler)
        result = sim_device.run(tcircs, shots=shots).result()


    all_counts = []
    for __, _id in enumerate(paulis):
        if _id == len(_id)*'I':
            all_counts.append({len(_id)*'0':shots})
        else:
            all_counts.append(result.get_counts(__))
    
    #compute the expectations
    expectations = []
    for i, count in enumerate(all_counts):
        #initiate the expectation value to 0
        expectation_val = 0
        #compute the expectation
        for el in count.keys():
            sign = 1
            #change sign if there are an odd number of ones
            if el.count('1')%2 == 1:
                sign = -1
            expectation_val += sign*count[el]/shots
        expectations.append(expectation_val)
    return expectations

def vqe(coeffs, loss_filename=None, params_filename=None, log_filename  = None, **kwargs):
    """
    Compute the VQE loss/energy.
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    loss_filename (String): Path to save file for VQE loss/energy.
    params_filename (String): Path to save file for VQE parameters.
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    
    Returns:
    (Float) VQE energy. 
    """
    start = timer()
    expectations = compute_expectations(**kwargs)
    loss = np.inner(coeffs, expectations)
    end = timer()
    if(log_filename is not None):
        print(f'Loss computed by VQE is {loss}, in {end - start} s.',file=open(log_filename, 'a'))
    
    if loss_filename is not None:
        with open(loss_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss])
    
    if params_filename is not None and parameters is not None:
        with open(params_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(parameters)
    return loss

def inner_circuit(ansatz, parameters1,parameters2):
    """
    Get the circuit used to compute inner products.
    ansatz (QuantumCircuit): The ansatz used
    parameters1 (Iterable[Float]): The first set of parameters
    parameters2 (Iterable[Float]): The second set of parameters
    Return 
    (QuantumCircuit): The circuit for the inner product
    """
    n_qubits  = ansatz.num_qubits
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qr, cr)
    
    q1=ansatz.assign_parameters(parameters1)
    q2=ansatz.assign_parameters(parameters2)
    
    qc.compose(q1.compose(q2.inverse()),qubits=range(0,n_qubits),inplace=True)
    qc.measure(qr,cr)
    return qc

def inner(shots, backend, noisy,ansatz,  seed_transpiler=13, **kwargs):
    """
    Compute the inner product between two ansatze with different parameters
    shots (Int): Number of VQE circuit execution shots.
    backend (IBM backend): The backend used to simulate the circuits
    noisy (Bool): True for noisy sim and False for ideal sim
    seed_transpiler (Int): The seed for the transpiler. Default is 13
    ansatz (QuantumCircuit): The ansatz to be used
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.

    Returns
    (Float) The computed inner product.
    """
    circuit = inner_circuit(ansatz=ansatz,**kwargs)
    if(not noisy):
        result = execute(circuit, backend=Aer.get_backend("qasm_simulator"), shots=shots).result()
    else:
        sim_device = AerSimulator.from_backend(backend)
        tcirc = transpile(circuit, backend, optimization_level=3, seed_transpiler=seed_transpiler)
        result = sim_device.run(tcirc, shots=shots).result()
    if(ansatz.num_qubits*'0' in result.get_counts().keys()):
        return result.get_counts()[ansatz.num_qubits*'0']/shots
    return 0

def vqd(coeffs, betas, paramList, loss_filename=None, params_filename=None,log_filename=None, **kwargs):
    """
    Compute the VQD loss.
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    betas (Iterable[Float]): betas for VQD
    paramList (Iterable[Iterable[Float]]) all prior parameters
    loss_filename (String): Path to save file for VQD loss.
    params_filename (String): Path to save file for VQD parameters.
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    
    Returns:
    (Float) VQD loss.
    """
    start = timer()
    loss=vqe(coeffs, loss_filename=None, params_filename=None, **kwargs)
    overlaps = [inner(shots=shots, backend=backend, noisy=noisy, ansatz=ansatz,parameters1=parameters,parameters2=p) for p in paramList]
    loss+=np.dot(betas, overlaps)
    end = timer()
    
    print(f'Loss computed by VQD is {loss}, in {end - start} s.',file=open(log_filename, 'a'))
    
    if loss_filename is not None:
        with open(loss_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss])
    
    if params_filename is not None and parameters is not None:
        with open(params_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(parameters)

    return loss



def vqe_cafqa_stim(inputs, coeffs, paulis,ansatz, loss_filename=None, params_filename=None, log_filename  = None):
    """
    Compute the CAFQA VQD loss/energy using stim.
    inputs (Dict): CAFQA VQE parameters (values in 0...3) as passed by hypermapper, e.g.: {"x0": 1, "x1": 0, "x2": 0, "x3": 2}
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    loss_filename (String): Path to save file for VQE loss.
    params_filename (String): Path to save file for VQE parameters.
    log_filename (String): Path to save file for VQE log.
    
    Returns:
    (Float) VQE loss.
    """
    start = timer()
    parameters = []
    # take the hypermapper parameters and convert them to vqe parameters
    for key in inputs:
        parameters.append(inputs[key]*(np.pi/2))
    vqe_qc = ansatz.assign_parameters(parameters)
    vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
    stim_qc = qiskit_to_stim(vqe_qc_trans)
    sim = stim.TableauSimulator()
    sim.do_circuit(stim_qc)
    pauli_expect = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]
    loss = np.dot(coeffs, pauli_expect)
    
    
    end = timer()
    if(log_filename is not None):
        print(f'Loss computed by CAFQA VQE is {loss}, in {end - start} s.',file=open(log_filename, 'a'))
    
    if loss_filename is not None:
        with open(loss_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss])
    
    if params_filename is not None and parameters is not None:
        with open(params_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(parameters)
    return loss

def vqd_cafqa_stim(inputs, coeffs, paulis,ansatz,betas,paramList, loss_filename=None, params_filename=None, log_filename  = None):
    """
    Compute the CAFQA VQD loss/energy using stim.
    inputs (Dict): CAFQA VQE parameters (values in 0...3) as passed by hypermapper, e.g.: {"x0": 1, "x1": 0, "x2": 0, "x3": 2}
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    betas (Iterable[Float]): betas for VQD
    paramList (Iterable[Iterable[Float]]) all prior parameters
    loss_filename (String): Path to save file for VQD loss.
    params_filename (String): Path to save file for VQD parameters.
    log_filename (String): Path to save file for VQD log.

    
    Returns:
    (Float) VQD loss.
    """
    start = timer()
    parameters = []
    # take the hypermapper parameters and convert them to vqe parameters
    for key in inputs:
        parameters.append(inputs[key]*(np.pi/2))
    vqe_qc = ansatz.assign_parameters(parameters)
    vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
    stim_qc = qiskit_to_stim(vqe_qc_trans)
    sim = stim.TableauSimulator()
    sim.do_circuit(stim_qc)
    pauli_expect = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]
    loss = np.dot(coeffs, pauli_expect)
    for i,v in enumerate(paramList):
        qc = inner_circuit(ansatz,parameters1 = parameters, parameters2 = v)
        qc.remove_final_measurements()
        qc_trans = transform_to_allowed_gates(qc)
        stim_qc = qiskit_to_stim(qc_trans)
        stim_qc.append("M", range(ansatz.num_qubits))
        sampler = stim_qc.compile_sampler()
        D=sampler.sample(shots=8912)
        loss += betas[i] * D.tolist().count([False] * ansatz.num_qubits)/8912
    end = timer()
    if(log_filename is not None):
        print(f'Loss computed by CAFQA VQD is {loss}, in {end - start} s.',file=open(log_filename, 'a'))
    
    if loss_filename is not None:
        with open(loss_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss])
    
    if params_filename is not None and parameters is not None:
        with open(params_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(parameters)
    return loss



