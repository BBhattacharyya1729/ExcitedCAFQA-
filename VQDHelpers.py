from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, ParameterVector, Parameter
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler.passes import RemoveBarriers
#from qiskit import transpile, execute
from qiskit_aer import AerSimulator, Aer
from timeit import default_timer as timer
import numpy as np
import csv
from functools import reduce
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


QNP_OR=QuantumCircuit(4)
QNP_OR.h([0,1])
QNP_OR.cx(0,2)
QNP_OR.cx(1,3)
phi=Parameter('φ')
QNP_OR.ry(phi,0)
QNP_OR.ry(phi,1)
QNP_OR.ry(phi,2)
QNP_OR.ry(phi,3)
QNP_OR.cx(0,2)
QNP_OR.cx(1,3)
QNP_OR.h([0,1])

QNP_PX = QuantumCircuit(4)
theta=Parameter('θ')
QNP_PX.cx(1,0)
QNP_PX.cx(3,1)
QNP_PX.cz(0,1)
QNP_PX.h(3)
QNP_PX.cx(3,2)
QNP_PX.ry(theta,2)
QNP_PX.ry(-theta,3)
QNP_PX.cz(0,3)
QNP_PX.cx(0,2)
QNP_PX.ry(-theta,3)
QNP_PX.ry(theta,2)
QNP_PX.cx(1,2)
QNP_PX.cx(1,3)
QNP_PX.ry(theta,3)
QNP_PX.ry(-theta,2)
QNP_PX.cx(0,2)
QNP_PX.cz(0,3)
QNP_PX.ry(theta,3)
QNP_PX.ry(-theta,2)
QNP_PX.cx(3,2)
QNP_PX.cx(1,3)
QNP_PX.h(3)
QNP_PX.cx(3,1)
QNP_PX.cx(1,0)

PI = QuantumCircuit(4)
PI.append(QNP_OR,[0,1,2,3])
PI=PI.assign_parameters([np.pi])
PI=PI.decompose()

Q = QuantumCircuit(4)
theta=Parameter('θ')
phi=Parameter('φ')
Q.append(PI,[0,1,2,3])
Q.append(QNP_PX,[0,1,2,3])
Q.append(QNP_OR,[0,1,2,3])
Q=Q.decompose()

A=QuantumCircuit(2)
theta=Parameter('θ')
phi=Parameter('φ')
A.cx(1,0)
A.rz(-phi-np.pi,1)
A.ry(-theta-np.pi/2,1)
A.cx(0,1)
A.ry(theta+np.pi/2,1)
A.rz(phi+np.pi,1)
A.cx(1,0)


def N_op(num_qubits):
    op = FermionicOp(
    {
        f"+_{i} -_{i}":1.0 for i in range(0,num_qubits)
    },
    num_spin_orbitals=num_qubits
    )
    
    mapper = JordanWignerMapper()
    qubitOp = mapper.map(op)
    return SparsePauliOp(coeffs=qubitOp.coeffs,data=[p[::-1] for p in qubitOp.paulis.to_labels()])

def N_alpha(num_qubits):
    op = FermionicOp(
    {
        f"+_{i} -_{i}":1.0 for i in range(0,num_qubits//2)
    },
    num_spin_orbitals=num_qubits
    )
    
    mapper = JordanWignerMapper()
    qubitOp = mapper.map(op)
    return qubitOp

def N_beta(num_qubits):
    op = FermionicOp(
    {
        f"+_{i} -_{i}":1.0 for i in range(num_qubits//2,num_qubits)
    },
    num_spin_orbitals=num_qubits
    )
    
    mapper = JordanWignerMapper()
    qubitOp = mapper.map(op)
    return qubitOp

def S_z(num_qubits):
    S_z = (N_alpha(num_qubits)-N_beta(num_qubits))/2
    return SparsePauliOp(coeffs=S_z.coeffs, data = [p[::-1] for p in S_z.paulis.to_labels()])


def S_plus(num_qubits):
    op = FermionicOp(
    {
        f"+_{i} -_{i+num_qubits//2}":1.0 for i in range(0,num_qubits//2)
    },
    num_spin_orbitals=num_qubits
    )
    
    mapper = JordanWignerMapper()
    qubitOp = mapper.map(op)
    return qubitOp


def S_minus(num_qubits):
    op = FermionicOp(
    {
        f"+_{i} -_{i-num_qubits//2}":1.0 for i in range(num_qubits//2,num_qubits)
    },
    num_spin_orbitals=num_qubits
    )
    
    mapper = JordanWignerMapper()
    qubitOp = mapper.map(op)
    return qubitOp

def S2(num_qubits):
    S2=S_plus(num_qubits) @ S_minus(num_qubits) + ((N_alpha(num_qubits)-N_beta(num_qubits))/2) @ ((N_alpha(num_qubits)-N_beta(num_qubits))/2-SparsePauliOp('I'*num_qubits))
    return SparsePauliOp(coeffs=S2.coeffs, data = [p[::-1] for p in S2.paulis.to_labels()])

def A_Ansatz(n_qubits,reps,initial=None):
    init=QuantumCircuit(n_qubits)
    if initial is not None:
        for i in range(len(initial)):
            if initial[i] == "1":
                init.x(i)
                
    t = ParameterVector('t', 2*reps*(n_qubits-2))
    qc= QuantumCircuit(n_qubits)
    current_index = 0
    for i in range(reps):
        if(n_qubits % 4 == 0):
            for n in range(n_qubits // 2 ):
                qc.append(A.assign_parameters([t[current_index],t[current_index+1]]),[2*n,2*n+1])
                current_index+=2
            for n in range(n_qubits // 2 - 1):
                if(n != n_qubits//4 -1):
                    qc.append(A.assign_parameters([t[current_index],t[current_index+1]]),[2*n+1,2*n+2])
                    current_index+=2
        else:
            for n in range(n_qubits // 2 ):
                if(n != (n_qubits+2)//4-1):
                    qc.append(A.assign_parameters([t[current_index],t[current_index+1]]),[2*n,2*n+1])
                    current_index+=2
            for n in range(n_qubits // 2 - 1):
                qc.append(A.assign_parameters([t[current_index],t[current_index+1]]),[2*n+1,2*n+2])
                current_index+=2
    return init.compose(qc.decompose()) 

def Q_Ansatz(n_qubits,reps,initial=None):
    init=QuantumCircuit(n_qubits)
    if initial is not None:
        for i in range(len(initial)):
            if initial[i] == "1":
                init.x(i)
    t = ParameterVector('t', reps*(n_qubits-2))
    qc= QuantumCircuit(n_qubits)
    current_index = 0
    for i in range(reps):
        for k in range(n_qubits//4):
            qc.append(Q.assign_parameters([t[current_index],t[current_index+1]]),[2*k,2*k+n_qubits//2,2*k+1,2*k+n_qubits//2+1])
            current_index+=2
        for k in range((n_qubits-2)//4):
            qc.append(Q.assign_parameters([t[current_index],t[current_index+1]]),[2*k+1,2*k+n_qubits//2+1,2*k+2,2*k+n_qubits//2+2])
            current_index+=2
    
    return init.compose(qc.decompose()) 


def getSDTab(stim_qc):
    N = stim_qc.num_qubits
    stabs = [stim.PauliString("_"*i + "Z"+"_"*(N-i-1)) for i in range(N)]
    destabs = [stim.PauliString("_"*i + "X"+"_"*(N-i-1)) for i in range(N)]
    stabs = [i.after(stim_qc) for i in stabs] 
    destabs = [i.after(stim_qc) for i in destabs] 
    return stabs,destabs

def getDecomp(op,stabs,destabs):
    s_ops = [stabs[i] for i,v in enumerate(destabs) if not v.commutes(op)]
    d_ops = [destabs[i] for i,v in enumerate(stabs) if not v.commutes(op)]
    
    s=stim.PauliString("I"*len(op))
    if(len(s_ops) > 0):
        s=reduce((lambda x, y: x * y), s_ops)
    
    d=stim.PauliString("I"*len(op))
    if(len(d_ops) > 0):
        d=reduce((lambda x, y: x * y), d_ops)
    a  = op.sign/(d * s).sign
    return a,d,s,s_ops,d_ops

def inner(stabs1,destabs1,stabs2,destabs2):
    marked = stabs1.copy()
    U_stabs = stabs1.copy()
    U_destabs = destabs1.copy()
    
    for op in stabs2:
        a,d,s,s_ops,d_ops = getDecomp(op,U_stabs,U_destabs)
        decomp  = s_ops+d_ops
        l=  [v for v in decomp if (v not in marked) ]
        
        if(len(l)==0):
            if(a==-1):
                return 0
        
        else:
            dk = l[0]
            index = U_destabs.index(dk)
            sk = U_stabs[index]
            U_destabs[index] = op
            for i,v in enumerate(U_stabs):
                if((not v.commutes(op)) and i!= index ):
                    U_stabs[i] =  U_stabs[i] * sk
                    marked[i] =  marked[i] * sk
            for i,v in enumerate(U_destabs):
                if((not v.commutes(op)) and i!= index ):
                    U_destabs[i] =  U_destabs[i] * sk

            marked.append(op)
            
    return 2**(len(marked[0])-len(marked))
            

def circuit_inner(circuit1,circuit2):
    stabs1,destabs1=getSDTab(circuit1)
    stabs2,destabs2=getSDTab(circuit2)
    return inner(stabs1,destabs1,stabs2,destabs2)