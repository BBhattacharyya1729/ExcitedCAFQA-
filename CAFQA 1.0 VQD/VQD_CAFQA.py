import numpy as np


from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter

from skquant.opt import minimize
import hypermapper
import json
import sys
from numbers import Number
from VQDHelpers import *

def molecule(atom_string, new_num_orbitals=None, charge=0, **kwargs):
    """
    Compute Hamiltonian for molecule in qubit encoding using Qiskit Nature.
    atom_string (String): string to describe molecule, passed to PySCFDriver.
    new_num_orbitals (Int): Number of orbitals in active space (if None, use default result from PySCFDriver).
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.

    Returns:
    (Iterable[Float], Iterable[String], String) (Pauli coefficients, Pauli strings, Hartree-Fock bitstring)
    """
    converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
    driver = PySCFDriver(
        atom=atom_string,
        basis="sto3g",
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
    qubitOp = converter.convert(ferOp, problem.num_particles)
    initial_state = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        converter
    )
    bitstring = "".join(["1" if bit else "0" for bit in initial_state._bitstr])
    # need to reverse order bc of qiskit endianness
    paulis = [x[::-1] for x in qubitOp.primitive.paulis.to_labels()]
    # add the shift as extra I pauli
    paulis.append("I"*len(paulis[0]))
    paulis = np.array(paulis)
    coeffs = list(qubitOp.primitive.coeffs)
    # add the shift (nuclear repulsion)
    coeffs.append(problem.nuclear_repulsion_energy)
    coeffs = np.array(coeffs).real
    return coeffs, paulis, bitstring

def BayesianOptimizer(fun,num_params,iterations,save_dir,name):
    """
    General purpose hypermapper based optimizer
    (Function) fun: Function to optimize over
    (Int) num_params: The number of parameters
    (Int) iterations: The number of iterations
    (String) save_dir: text name of the save directory
    (String) name: name for all the logging
    """
    
    hypermapper_config_path = save_dir + "/"+name+"_hypermapper_config.json"
    config = {}
    config["application_name"] = "cafqa_optimization_"+name
    config["optimization_objectives"] = ["value"]
    config["design_of_experiment"] = {}
    config["design_of_experiment"]["number_of_samples"] = iterations
    config["optimization_iterations"] = iterations
    config["models"] = {}
    config["models"]["model"] = "random_forest"
    config["input_parameters"] = {}
    config["print_best"] = True
    config["print_posterior_best"] = True
    for i in range(num_params):
        x = {}
        x["parameter_type"] = "ordinal"
        x["values"] = [0, 1, 2, 3]
        x["parameter_default"] = 0
        config["input_parameters"]["x" + str(i)] = x
    config["log_file"] = save_dir + '/'+name+'_hypermapper_log.log'
    config["output_data_file"] = save_dir + "/"+name+"_hypermapper_output.csv"
    with open(hypermapper_config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)
    stdout=sys.stdout
    with open(save_dir+"/"+name+'_optimizer_log.txt', 'w') as sys.stdout:
        hypermapper.optimizer.optimize(hypermapper_config_path,fun)
    sys.stdout = stdout
    
    fun_ev = np.inf
    x = None
    with open(config["log_file"]) as f:
        lines = f.readlines()
        counter = 0
        for idx, line in enumerate(lines[::-1]):
            if line[:16] == "Best point found" or line[:29] == "Minimum of the posterior mean":
                counter += 1
                parts = lines[-1-idx+2].split(",")
                value = float(parts[-1])
                if value < fun_ev:
                    fun_ev = value
                    x = [int(y) for y in parts[:-1]]
            if counter == 2:
                break
    return fun_ev, x


def run_cafqa_vqe(coeffs,paulis,ansatz,iterations,save_dir,name):
    num_params = ansatz.num_parameters
    loss_path =  save_dir + "/"+name+"_loss.txt"
    params_path = save_dir + "/"+name+"_params.txt"
    log_path = save_dir + "/"+name+"_log.txt"
    energy,parameters = BayesianOptimizer(fun = lambda inputs: vqe_cafqa_stim(inputs=inputs, 
                                                                              coeffs=coeffs, 
                                                                              paulis=paulis,
                                                                              ansatz=ansatz, 
                                                                              loss_filename=loss_path, 
                                                                              params_filename=params_path, 
                                                                              log_filename  = log_path),
                                          num_params=num_params,iterations=iterations,save_dir=save_dir,name=name)
    return energy,np.array(parameters)*np.pi/2


def run_cafqa_vqd(coeffs,paulis,ansatz,iterations,save_dir,name,k):
    num_params = ansatz.num_parameters
    beta = np.sum(np.abs(coeffs))*2
    energies=[]
    betas=[]
    paramList=[]
    for i in range(k):
        loss_path =  save_dir + "/"+name+" Energy Level: "+str(i)+"_loss.txt"
        params_path = save_dir + "/"+name+" Energy Level: "+str(i)+"_params.txt"
        log_path = save_dir + "/"+name+" Energy Level: "+str(i)+"_log.txt"
        loss,parameters = BayesianOptimizer(fun = lambda inputs: vqd_cafqa_stim(inputs=inputs, 
                                                                                  coeffs=coeffs, 
                                                                                  paulis=paulis,
                                                                                  ansatz=ansatz, 
                                                                                  loss_filename=loss_path, 
                                                                                  params_filename=params_path, 
                                                                                  log_filename  = log_path,
                                                                                  betas=betas,
                                                                                  paramList=paramList),
                                              num_params=num_params,iterations=iterations,save_dir=save_dir,name=name+" Energy Level: "+str(i))
        d={"x"+str(i):v for i,v in enumerate(parameters)}
        energies.append(vqe_cafqa_stim(inputs=d,coeffs=coeffs,paulis=paulis,ansatz=ansatz))
        paramList.append(np.array(parameters)*np.pi/2)
        betas.append(beta)
    return energies,paramList
        