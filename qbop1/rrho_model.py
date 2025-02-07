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


"""Computes thermal effects using the rigid-rotor 
   harmonic oscillator approximation
   constants :: physical constants 
   translational :: translational degrees of freedom contributions
   electronic :: electronic degrees of freedom contributions
   rotational :: rotational degrees of freedom contributions
   q_total :: total system partition functions
   thermal :: total thermal corrections
   """

import numpy as np
from scipy.constants import physical_constants

def constants(type_const, unit_system):
    """Values of Boltzmann's constant (k_B), ideal gas constant (R),
       and Planck's constant (h) with the necessary units
       
       
       Parameters
       ----------
       type_const: :obj:`numpy.ndarray`
           a 1d-array containing the math symbols of the constants
       unit_system: :obj:`numpy.ndarray`
           a 1d-array containing the units (either 'kcal/mol' or 'SI')
       
       Returns
       -------
       data: :obj:`tuple`
           value of each constant
    """
    
    data = []
    size = type_const.shape[0]
    
    values = {'k_B':{'SI':physical_constants['Boltzmann constant']},
              'R':{'kcal/mol': (8.314472 * 627.5098)/ (2625.5002e3)},
              'h':{'SI':physical_constants['Planck constant']}}
    for l in range(size):
        const = type_const[l]
        units = unit_system[l]
        if const in ['k_B','h']:
            value, unit, uncertainty = values[const][units]
        elif const == 'R':
            value = values[const][units]
        data += value,
    
    return data

def translational(total_mass, T, P):
    """Calculate the translational contributions
    
       Parameters
       ----------
       total_mass: :obj:`numpy.float64`
           total mass of the molecule or atom
       T: :obj:`numpy.float64`
           temperature of the system (in Kelvin)
       P: :obj:`numpy.float64`
           pressure of the system (in atmosphere)
       
       Returns
       -------
       E_t: :obj:`numpy.float64`
           thermal correction from translation (in kcal/mol)
       q_t: :obj:`numpy.float64`
           partition function from translation
       S_t: :obj:`numpy.float64`
           entropy correction from translation (in kcal/mol*K)
       Cv_t: :obj:`numpy.float64`
           heat capacity at constant volume 
           from translation (in kcal/mol*K)
       """
    
    k_B, R, h = constants(np.array(['k_B','R','h']),np.array(['SI','kcal/mol','SI']))
    P_Pa = P * 101325  # atmopshere to Pascal
    mass = total_mass * 1.6605402e-27 # amu to kg (from IUPAC)
    
    E_t = (3/2) * R * T
    q_t = (np.power(2 * np.pi * mass * k_B * T / (h**2),3/2)) * (k_B * T / P_Pa)
    S_t = R * (np.log(q_t) + 5/2)
    Cv_t = (3/2) * R
    return (E_t, q_t, S_t, Cv_t)

def electronic(multiplicity):
    """Calculate the electronic contributions
    
       Parameters
       ----------
       multiplicity: :obj:`np.int64`
           2S+1, where S is the number of unpaired spin
           states in a molecule
       
       Returns
       -------
       E_e: :obj:`numpy.float64`   
           thermal correction from electronic (in kcal/mol)
       q_e: :obj:`numpy.float64`
           partition function from electronic
       S_e: :obj:`numpy.float64`
           entropy correction from electronic (in kcal/mol*K)
       Cv_e: :obj:`numpy.float64`
           heat capacity at constant volume 
           from electronic (in kcal/mol*K)"""
    
    R = constants(np.array(['R']),np.array(['kcal/mol']))[0]
    E_e = 0
    q_e = multiplicity
    S_e = R * np.log(q_e)
    Cv_e = 0
    return (E_e, q_e, S_e, Cv_e)

def rotational(atoms, rot_constant, symmetry_numb, T):
    """Calculate the electronic contributions
    
       Parameters
       ----------
       rot_constant: :obj:`numpy.ndarray` or `numpy.float64`
           1d array or scalar rotational constants (in Joules)
       symmetry_numb: :obj:`int`
           the rotational symmetry number
       T: :obj:`numpy.float64`
           temperature in K
       
       Returns
       -------
       E_r: :obj:`numpy.float64`     
           thermal correction from rotational (in kcal/mol)
       q_r: :obj:`numpy.float64`
           partition function from rotational
       S_r: :obj:`numpy.float64`
           entropy correction from rotational (in kcal/mol*K)
       Cv_r: :obj:`numpy.float64`
           heat capacity at constant volume 
           from rotational (in kcal/mol*K)"""
    
    R, k_B = constants(np.array(['R','k_B']),np.array(['kcal/mol','SI']))
    size = rot_constant.shape[0]
    
    if atoms.shape[0] == 1:
        E_r = 0
        q_r = 1
        S_r = 0
        Cv_r = 0
    elif size == 2:
        E_r = R * T
        q_r = (1 / symmetry_numb) * np.prod(np.sqrt(k_B * T / rot_constant))
        S_r = R * (np.log(q_r) + 1)
        Cv_r = R
    elif size == 3:
        E_r = (3 / 2) * R * T
        q_r = (np.sqrt(np.pi) / symmetry_numb) * np.prod(np.sqrt(k_B * T / rot_constant))
        S_r = R * (np.log(q_r) + 3/2)
        Cv_r = (3 / 2) * R
    return (E_r, q_r, S_r, Cv_r)

def q_total(q_t, q_e, q_r, q_v_bot, q_v_0):
    """Calculate the total partition function 
       starting at the bottom-of-the-well and 
       from the first excited vibrational state
    
       Parameters
       ----------
       q_t: :obj:`numpy.float64`
           translational partion function
       q_e: :obj:`numpy.float64`
           electronic partition function
       q_r: :obj:`numpy.float64`
           rotational partition function
       q_v_bot: :obj:`numpy.float64`
           vibrational partition function 
           starting from ground-state
           (i.e., bottom-of-the-well)
       q_v_0: :obj:`numpy.float64`
           vibrational partition function 
           starting from first excited state
           
       Returns
       -------
       q_bot_tot: :obj:`numpy.float64`
            total partition function
            from ground state 
            (i.e., bottom-of-the-well)
       q_v0_tot: :obj:`numpy.float64`
            total partition function starting
            from first excited state
    """
        
    q_bot_tot = q_t * q_e * q_r * q_v_bot
    q_v0_tot = q_t * q_e * q_r * q_v_0
    return (q_bot_tot, q_v0_tot)

def thermal(atoms, total_mass, mult, rot_constant, 
            symmetry_numb, T, P, qbop_info):
    """Calculate thermal energy, partition function,
       entropy, and heat capacity at constant volume
    
       Parameters
       ----------
       atoms: :obj:`numpy.ndarray`
           1d-array containing the 
           atom symbols
       total_mass: :obj:`numpy.float64`
           total mass of the molecule 
           or atom (in atomic units)
       mult: :obj:`numpy.int64`
           2S+1, where S is the number of unpaired spin
           states in a molecule
       rot_constant: :obj:`numpy.ndarray`
           the rotational constant of 
           the molecule (in J)
       T: :obj:`float`
           temperature in Kelvin
       P: :obj:`numpy.float64`
           pressure in atmosphere 
       qbop_info: :obj:`numpy.ndarray`
           vibrational partition functions 
           (either from QBOP or traditional method)
           
       Returns
       -------
       thermal_contr: :obj:`numpy.ndarray`
            thermal contributions from translational, 
            electronic, rotational, and vibrational motions
       q_contr: :obj:`numpy.ndarray`
            partion functions from translational, 
            electronic, rotational, and vibrational motions
       entropy_contr: :obj:`numpy.ndarray`
            entropy contributions from translational, 
            electronic, rotational, and vibrational motions
       Cv: :obj:`numpy.ndarray`
            heat capacity at constant volume from translational, 
            electronic, rotational, and vibrational motions
       """
    
    E_t, q_t, S_t, Cv_t = translational(total_mass,T,P) 
    E_e, q_e, S_e, Cv_e = electronic(mult)
    E_r, q_r, S_r, Cv_r = rotational(atoms, rot_constant, symmetry_numb, T)
    E_v, q_v_bot, q_v_0, S_v, Cv_v = qbop_info
    q_bot_tot, q_v0_tot = q_total(q_t, q_e, q_r, q_v_bot, q_v_0)

    thermal_contr = np.array([E_e, E_t, E_r, E_v])
    q_contr = np.array([q_bot_tot, q_v0_tot, q_v_bot, q_v_0, q_e, q_t, q_r])
    entropy_contr = np.array([S_e, S_t, S_r, S_v]) 
    Cv = np.array([Cv_e, Cv_t, Cv_r, Cv_v])
    return (thermal_contr, q_contr, entropy_contr, Cv) 