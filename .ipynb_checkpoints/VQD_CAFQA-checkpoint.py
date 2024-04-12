from __future__ import print_function
import numpy as np


from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
import qiskit_nature.settings
qiskit_nature.settings.use_pauli_sum_op = False
from skquant.opt import minimize
from VQDHelpers import *
from qiskit.circuit import QuantumCircuit,Parameter,ParameterVector

def molecule(atom_string, new_num_orbitals=None, charge=0, basis="sto3g", **kwargs):
    """
    Compute Hamiltonian for molecule in qubit encoding using Qiskit Nature.
    atom_string (String): string to describe molecule, passed to PySCFDriver.
    new_num_orbitals (Int): Number of orbitals in active space (if None, use default result from PySCFDriver).
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.

    Returns:
    (Iterable[Float], Iterable[String], String) (Pauli coefficients, Pauli strings, Hartree-Fock bitstring)
    """
    driver = PySCFDriver(
        atom=atom_string,
        basis=basis,
        charge=charge,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    if new_num_orbitals is not None:
        num_electrons = (problem.num_alpha, problem.num_beta)
        transformer = ActiveSpaceTransformer(num_electrons, new_num_orbitals)
        problem = transformer.transform(problem)
    ferOp = problem.hamiltonian.second_q_op()
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubitOp = mapper.map(ferOp)
    initial_state = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper
    )
    bitstring = "".join(["1" if bit else "0" for bit in initial_state._bitstr])
    # need to reverse order bc of qiskit endianness
    paulis = [x[::-1] for x in qubitOp.paulis.to_labels()]
    # add the shift as extra I pauli
    paulis.append("I"*len(paulis[0]))
    paulis = np.array(paulis)
    coeffs = list(qubitOp.coeffs)
    # add the shift (nuclear repulsion)
    coeffs.append(problem.nuclear_repulsion_energy)
    coeffs = np.array(coeffs).real
    return coeffs, paulis, bitstring


def JWmolecule(atom_string, new_num_orbitals=None, charge=0, basis="sto3g", **kwargs):
    """
    Compute Hamiltonian for molecule in qubit encoding using Qiskit Nature.
    atom_string (String): string to describe molecule, passed to PySCFDriver.
    new_num_orbitals (Int): Number of orbitals in active space (if None, use default result from PySCFDriver).
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.

    Returns:
    (Iterable[Float], Iterable[String], String) (Pauli coefficients, Pauli strings, Hartree-Fock bitstring)
    """
    driver = PySCFDriver(
        atom=atom_string,
        basis=basis,
        charge=charge,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    if new_num_orbitals is not None:
        num_electrons = (problem.num_alpha, problem.num_beta)
        transformer = ActiveSpaceTransformer(num_electrons, new_num_orbitals)
        problem = transformer.transform(problem)
    ferOp = problem.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    qubitOp = mapper.map(ferOp)
    initial_state = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper
    )
    bitstring = "".join(["1" if bit else "0" for bit in initial_state._bitstr])
    # need to reverse order bc of qiskit endianness
    paulis = [x[::-1] for x in qubitOp.paulis.to_labels()]
    # add the shift as extra I pauli
    paulis.append("I"*len(paulis[0]))
    paulis = np.array(paulis)
    coeffs = list(qubitOp.coeffs)
    # add the shift (nuclear repulsion)
    coeffs.append(problem.nuclear_repulsion_energy)
    coeffs = np.array(coeffs).real
    return coeffs, paulis, bitstring



def vqe_genetic_cafqa_stim(inputs, coeffs, paulis,ansatz, loss_filename=None, params_filename=None, log_filename  = None):
    """
    Compute the CAFQA VQD loss/energy using stim.
    inputs (Iterable[int]): CAFQA VQE parameters (values in 0...3)
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    loss_filename (String): Path to save file for VQE loss.
    params_filename (String): Path to save file for VQE parameters.
    log_filename (String): Path to save file for VQE log.
    
    Returns:
    (Float) VQE loss.
    """
    start = timer()
    parameters = np.array(inputs) * np.pi/2
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


def vqd_genetic_cafqa_stim(inputs, coeffs, paulis,ansatz,betas,paramList,ansatzList, loss_filename=None, params_filename=None, log_filename  = None):
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
    parameters = np.array(inputs) * np.pi/2
    vqe_qc = ansatz.assign_parameters(parameters)
    vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
    stim_qc = qiskit_to_stim(vqe_qc_trans)
    sim = stim.TableauSimulator()
    sim.do_circuit(stim_qc)
    pauli_expect = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]
    loss = np.dot(coeffs, pauli_expect)
    for i,v in enumerate(paramList):        
        vqe_qc = ansatz.assign_parameters(parameters)
        vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
        stim_qc = qiskit_to_stim(vqe_qc_trans)
        
        vqe_qc1 = ansatzList[i].assign_parameters(v)
        vqe_qc1_trans = transform_to_allowed_gates(vqe_qc1)
        stim_qc1 = qiskit_to_stim(vqe_qc1_trans)
        
        loss += betas[i] * circuit_inner(stim_qc1,stim_qc)


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


import pygad

def CliffordGeneticOptimizer(func,
                             num_params,
                             budget=1000,
                             num_parents_mating = 4, 
                             sol_per_pop = 8, 
                             parent_selection_type = "sss", 
                             keep_parents = 1, 
                             crossover_type = "single_point",
                             crossover_probability=0.75, 
                             mutation_type = "random", 
                             mutation_probability = 0.01,
                             keep_elitism=1,
                            parallel = 4):
    def fitness_func(ga_instance, solution, solution_idx):
        return -1*func(solution)
    
    ga_instance = pygad.GA(num_generations=budget,
                                num_parents_mating=num_parents_mating,
                                fitness_func=fitness_func,
                                sol_per_pop=sol_per_pop,
                                num_genes=num_params,
                                parent_selection_type=parent_selection_type,
                                keep_parents=keep_parents,
                                crossover_type=crossover_type,
                                mutation_type=mutation_type,
                                gene_space=[0,1,2,3],
                                crossover_probability=crossover_probability,
                                mutation_probability=mutation_probability,
                                keep_elitism=keep_elitism,
                                parallel_processing=parallel
                              )
    ga_instance.run()
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    return -1*solution_fitness,solution

def run_genetic_cafqa_vqe(coeffs,paulis,ansatz,save_dir=None,name=None,
                        budget=1000,
                         num_parents_mating = 4, 
                         sol_per_pop = 8, 
                         parent_selection_type = "sss", 
                         keep_parents = 1, 
                         crossover_type = "single_point",
                         crossover_probability=0.75, 
                         mutation_type = "random", 
                         mutation_probability = 0.01,
                         keep_elitism=1,
                         parallel=4):
    
    if(save_dir is not None and name is not None):
        loss_path =  save_dir + "/"+name+"_loss.txt"
        params_path = save_dir + "/"+name+"_params.txt"
        log_path = save_dir + "/"+name+"_log.txt"
    else:
        loss_path =  None
        params_path = None
        log_path = None
    

    e,state=CliffordGeneticOptimizer(lambda x: vqe_genetic_cafqa_stim(x, coeffs, paulis,ansatz, loss_filename=loss_path, params_filename=params_path, log_filename  = log_path),
                             ansatz.num_parameters,
                             budget=budget,
                             num_parents_mating = num_parents_mating, 
                             sol_per_pop = sol_per_pop, 
                             parent_selection_type = parent_selection_type, 
                             keep_parents = keep_parents, 
                             crossover_type = crossover_type,
                             crossover_probability=crossover_probability, 
                             mutation_type = mutation_type, 
                             mutation_probability = mutation_probability,
                             keep_elitism=keep_elitism,
                            parallel=parallel)
    return np.array(state) * np.pi/2, e

def run_genetic_cafqa_vqd(coeffs,paulis,ansatzList,k,save_dir=None,name=None,
                        budget=1000,
                         num_parents_mating = 4, 
                         sol_per_pop = 8, 
                         parent_selection_type = "sss", 
                         keep_parents = 1, 
                         crossover_type = "single_point",
                         crossover_probability=0.75, 
                         mutation_type = "random", 
                         mutation_probability = 0.01,
                         keep_elitism=1,
                         parallel=4):
    
    beta = np.sum(np.abs(coeffs))*2
    energies=[]
    betas=[]
    paramList=[]
    ansatz_list = []
    
    
    for i in range(k):
        
        if(save_dir is not None and name is not None):
            loss_path =  save_dir + "/"+name+" Energy Level: "+str(i)+"_loss.txt"
            params_path = save_dir + "/"+name+" Energy Level: "+str(i)+"_params.txt"
            log_path = save_dir + "/"+name+" Energy Level: "+str(i)+"_log.txt"
        else:
            loss_path =  None
            params_path = None
            log_path = None
        
        
        
        e,state=CliffordGeneticOptimizer(lambda x: vqd_genetic_cafqa_stim(x, coeffs, paulis,ansatzList[i],betas,paramList,ansatz_list,loss_filename=loss_path, params_filename=params_path, log_filename  = log_path),
                             ansatzList[i].num_parameters,
                             budget=budget,
                             num_parents_mating = num_parents_mating, 
                             sol_per_pop = sol_per_pop, 
                             parent_selection_type = parent_selection_type, 
                             keep_parents = keep_parents, 
                             crossover_type = crossover_type,
                             crossover_probability=crossover_probability, 
                             mutation_type = mutation_type, 
                             mutation_probability = mutation_probability,
                             keep_elitism=keep_elitism,
                            parallel=parallel)
        loss=vqe_genetic_cafqa_stim(inputs=state, coeffs=coeffs, paulis=paulis,ansatz=ansatzList[i])
        energies.append(loss)
        ansatz_list.append(ansatzList[i])
        paramList.append(np.array(state)*np.pi/2)
        betas.append(beta)
    return energies,paramList