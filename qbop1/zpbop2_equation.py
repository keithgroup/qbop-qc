# MIT License
# 
# Copyright (c) 2025, Barbaro Zulueta
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Computes the zpve energy contributions
   E_harm :: two-body EHT harmonic energy
   conditional_statements :: conditional protocols
   E_anharm :: short-range anharmonic energy terms
   calc_cosines :: calculate cosines using the law of cosines
   three_body :: three-coupled harmonic oscillator term 
   three_energy_decomp :: decompose three-body energy into individual two-bodies
   zpe :: total zpve energy in kcal/mol along with regular and effective two-body energies
   """

import numpy as np
from . import parameters as param

def E_harm(bo, beta):
    """Calculate the harmonic zpve energy from extended-Hückel theory
    
       Parameters
       ----------
       bo: :obj:`numpy.float64`
           Mulliken bond order
       beta: :obj:`numpy.float64`
           Extend-Hückel theory parameter
           for Mulliken bond order
       
       Returns
       -------
       zpve: :obj:`numpy.float64` 
           total EHT harmonic energy
       """
    
    zpve = 2 * beta * np.abs(bo)
    return zpve

def conditional_statements(pair_atoms, bo, size):
    """Condition protocols for zpve
    
       Parameters
       ----------
       pair_atoms: :obj:`numpy.ndarray`
           array showing the two pair of atoms
       bo: :obj:`numpy.float64`
           Mulliken bond order
       size :obj:`numpy.int64`
           Number of atoms in molecule
           
       Returns
       -------
       corrections: :obj:`int`
           the correction value: 
               True -> 0 > bo
               False -> bo > 0
       """
    
    # Conditional protocols for computing zpve bond energies
    if 0 > bo and size > 2:
        corrections = True # anti-bonding 
    else:
        corrections = False # bonding
    return corrections

def E_anharm(distance, pre_exp, zeta, R_param):
    """Compute the short-range anharmonic interactions
    
       Parameters
       ----------
       distance: :obj:`numpy.float64`
           pair-wise distances
       pre_exp: :obj:`numpy.float64`
           negative pre-exponential for the decaying exponential
       zeta :obj:`numpy.float64`
           exponent of the decaying potential (units:A^-1)
       R_param :obj:`numpy.float64`
           classical turning point distance (units:A)
           
       Returns
       -------
       value: :obj:`numpy.float64`
           negative short-range anharmonic energy
    """
    
    if pre_exp == 'None':
        value = 0
    else:
        value = pre_exp * np.exp(-zeta * (distance - R_param))
    return value

def calc_cosines(R_ij, R_ik, R_jk):
    """Calculate cosines using the law of cosines
    
       Parameters
       ----------
       R_ij: :obj:`numpy.float64`
           distances of atoms ij
       R_ik: :obj:`numpy.float64`
           distances of atoms ik
       R_jk: :obj:`numpy.float64`
           distances of atoms jk
           
       Returns
       -------
       cosines: :obj:`numpy.ndarray`
           array with values of cos_ij, cos_ik, and cos_jk 
    """
    
    cos_ij = ((R_ik**2 + R_jk**2) - R_ij**2)/(2 * R_ik * R_jk)
    if cos_ij > 1:
        cos_ij = 1
    if cos_ij < -1:
        cos_ij = -1

    # Cosine between atoms i and k
    cos_ik = ((R_ij**2 + R_jk**2) - R_ik**2)/(2 * R_ij * R_jk)
    if cos_ik > 1:
        cos_ik = 1
    if cos_ik < -1:
        cos_ik = -1
        
    # Cosine between atoms j and k
    cos_jk = ((R_ij**2 + R_ik**2) - R_jk**2)/(2 * R_ij * R_ik)
    if cos_jk > 1:
        cos_jk = 1
    if cos_jk < -1:
        cos_jk = -1
    
    cosines = np.array([cos_ij, cos_ik, cos_jk])
    return cosines

def three_body(kappas, distances, bo):
    """Calculate three-coupled harmonic oscillators
    
       Parameters
       ----------
       kappas: :obj:`numpy.ndarray`
           intrinsic two-body parameters in a three-body typology
       distances: :obj:`numpy.ndarray`
           distances for atom pairs ij, ik, and jk
       bo: :obj:`numpy.ndarray`
           bond orders for atom pairs ij, ik, and jk
           
       Returns
       -------
       value: :obj:`numpy.float64`
           three-body coupled energy
    """
    
    cosines = calc_cosines(*distances)
    value = np.prod(kappas) * np.prod(2 * np.abs(bo)) * np.prod(cosines)
    return value

def three_energy_decomp(three_two_decomp, three_body_value, two_bodies, pairs):
    """Calculate the effective two-body energies from three-body energy
    
       Parameters
       ----------
       three_two_decomp: :obj:`numpy.ndarray`
           effective two body energies for ij, ik, and jk decomposed from three-body
       three_body_value: :obj:`numpy.ndarray`
           three-body energy between atoms i,j and k
       two_bodies: :obj:`numpy.ndarray`
           two body energies for atom pairs ij, ik, and jk
       pairs: :obj:`numpy.ndarray`
           indexes of each pair of atoms
           
       Returns
       -------
       three_two_decomp: :obj:`numpy.ndarray`
           effective two body energies for ij, ik, and jk decomposed from three-body
    """
    
    # two body gross contributions
    two_ij = two_bodies[pairs[0]]
    two_ik = two_bodies[pairs[1]]
    two_jk = two_bodies[pairs[2]]
    sum_two = two_ij + two_ik + two_jk
    
    # three-body contributions decomposed into two bodies
    three_two_decomp[pairs[0]] += three_body_value * (two_ij / sum_two)
    three_two_decomp[pairs[1]] += three_body_value * (two_ik / sum_two)
    three_two_decomp[pairs[2]] += three_body_value * (two_jk / sum_two)
    return three_two_decomp

def zpe(atoms, bo, distance_matrix, parameter_folder):
    """Calculate ZPE-BOP2 energies and regular and effective two-bodies
    
       Parameters
       ----------
       atoms: :obj:`numpy.ndarray`
           N array showing the atoms
       bo: :obj:`numpy.ndarray`
           Mulliken bond order
       distance_matrix: :obj:`numpy.ndarray`
           distance matrix
       parameter_folder: :obj:`str`
           name of the path/folder where ZPE-BOP2 data is located

           
       Returns 
       -------
       zpve: :obj:`numpy.float64`
           the total zpve in kcal/mol
       two_bodies: :obj:`numpy.ndarray`
           NxN matrix containing EHT energy terms
       three_bodies: :obj:`numpy.ndarray`
           NxN matrix containing the effective two-body energies
       """
    
    size = atoms.shape[0]
    harmonic = np.zeros((size,size), dtype = np.float64) # harmonic contributions
    anharmonic = np.zeros((size,size), dtype = np.float64) # short-range anharmonics
    three_two_decomp = np.zeros((size,size), dtype = np.float64) # three-body
    
    # two-body contributions
    for l in range(1,size):
        for n in range(l):
            # important properties
            index = (l,n)
            pair_atoms = np.array([atoms[l], atoms[n]])
            bo_pair = bo[l][n]
            distance = distance_matrix[l][n]
            
            anti_corr = conditional_statements(pair_atoms, bo_pair, size)
            
            # fitting parameters
            beta, pre_exp, zeta, R_param = param.zpe_pair(pair_atoms, parameter_folder, anti_corr)
            

            # extended Hückel covalent bonding for sigma and pi bonding 
            harmonic[l][n] = E_harm(bo_pair, beta)
            anharmonic[l][n] = E_anharm(distance, pre_exp, zeta, R_param)
                       
    # Total Pairwise Interactions
    two_bodies = harmonic + anharmonic
    
    # three-body contributions
    three_energy = 0
    n = 0
    bonds = np.array(['C~C','N~N','O~O','B~B','B~N',
                      'N~B','O~B','B~O','B~C','C~B',
                      'C~O','O~C','C~N','N~C','N~O',
                      'O~N'])
    for i in range(2,size):
        for j in range(1,i):
            for k in range(j):
                    bond_ij = f'{atoms[i]}~{atoms[j]}'
                    bond_ik = f'{atoms[i]}~{atoms[k]}'
                    bond_jk = f'{atoms[j]}~{atoms[k]}'
                    three_bonds = np.array([bond_ij, bond_ik, bond_jk])
                    if np.prod(np.isin(three_bonds,bonds)) == 1:
                        n += 1
                        pairs = [(i,j),(i,k),(j,k)]
                        distances = np.array([distance_matrix[l] for l in pairs])
                        pair_atoms = np.array([[atoms[l],atoms[z]] for l, z in pairs])
                        bo_pair = np.array([bo[l] for l in pairs])
                        anti_corr = np.array([conditional_statements(pair_atoms[i], bo_pair[i], size) for i in range(3)])
                        kappas = np.array([param.zpe_pair(pair_atoms[l], parameter_folder, anti_corr[l], two_body = False) for l in range(3)])
                        three_body_value = three_body(kappas, distances, bo_pair)
                        three_energy += three_body_value
                        three_two_decomp = three_energy_decomp(three_two_decomp, three_body_value, two_bodies, pairs)
                    else:
                        continue
                        
    zpve = np.sum(two_bodies) + three_energy
    return (zpve, two_bodies, three_two_decomp) 