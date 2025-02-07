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

"""Compute the necessary spatial contributions
   distances :: Compute the distance matrix
   masses :: computes the total mass and individual atom masses
   convert :: convert au units to si units
   center_mass :: compute center of mass
   moment_of_inertia :: moment of inertia
   principal_axis :: compute principal axis of rotations
   B :: compute the rotational constants (assuming rigid rotor approximation)
   vib_degrees_freedom :: number of vibrational degrees of freedoms
   point_group_to_sigma :: determine the rotational symmetry number from point group
   all_data :: return all spatial-dependent data"""

import numpy as np
import sys

def distances(XYZ):
    """Compute the distance matrix
    
       Parameters
       ----------
       XYZ: obj:`numpy.ndarray`
           Standard orientation cartesian coordinates.
       
       Returns
       -------
       distance_matrix: obj:`numpy.ndarray`
           Calculated distance matrix (should agree with Gaussian16 output).
    """
    
    length = XYZ.shape[0] 
    distance_matrix = np.zeros((length,length), dtype=np.float64)
        
    for i in range(1,length):
        for j in range(i):
            DeltaX = XYZ[j][0] - XYZ[i][0]
            DeltaY = XYZ[j][1] - XYZ[i][1]
            DeltaZ = XYZ[j][2] - XYZ[i][2]
            distance_matrix[i][j] = np.sqrt(DeltaX**2 + DeltaY**2 + DeltaZ**2)
    return distance_matrix

def masses(atoms):
    """Calculate the total mass and get the mass of each atom 
           
       Parameters
       ----------
       atoms: obj:`numpy.ndarray`
           Atomic symbols
       
       Returns
       -------
       total_mass: obj:`numpy.float64`
           Sum of all atom masses (in g/mol)
       masses: obj:`numpy.ndarray`
           Array of atom masses (in g/mol)
       """
    
    mass = {'H':1.00783,'He':4.0026,'Li':7.01600,'Be':9.01218,
            'B':11.00931,'C':12.00000,'N':14.00307,'O':15.99491,
            'F':18.99840,'Ne':20.180,'Na':22.98977,'Mg':23.98504,
            'Al':26.98154,'Si':27.97693,'P':30.97376,'S':31.97207,
            'Cl':34.96885,'Ar':39.948}
    
    masses = np.array([mass[i] for i in atoms])
    total_mass = np.sum(masses)
    return (total_mass, masses)

def convert(xyz, masses):
    """Convert from AU to SI units for individual atom cartesians and masses
               
       Parameters
       ----------
       xyz: obj:`numpy.ndarray`
           Standard orientation cartesian coordinates (in A)
       masses: obj:`numpy.ndarray`
           Array atom masses (in g/mol)
       
       Returns
       -------
       xyz_si: obj:`numpy.ndarray`
           Standard orientation cartesian coordinates (in m)
       masses_si: obj:`numpy.ndarray'
           Array of atom masses (in kg/mol)
    """
    xyz_si = xyz * 1e-10 # in meters
    masses_si = masses * 1.66053906660e-27 # in kg
    return (xyz_si, masses_si)

def center_mass(masses, xyz):
    """Compute center of mass
               
       Parameters
       ----------
       xyz: obj:`numpy.ndarray`
           Standard orientation cartesian coordinates
       masses: obj:`numpy.ndarray`
           Array atom masses
       
       Returns
       -------
       R_CM: obj:`numpy.ndarray`
           Standard orientation cartesian coordinates of the center of mass
    """
    
    R_CM = np.dot(masses,xyz) / np.sum(masses)
    return R_CM

def moment_of_inertia(xyz, R_CM, masses):
    """Moment of inertia
                   
       Parameters
       ----------
       xyz: obj:`numpy.ndarray`
           Standard orientation cartesian coordinates
       R_CM: obj:`numpy.ndarray`
           Standard orientation cartesian coordinates of the center of mass
       masses: obj:`numpy.ndarray`
           Array atom masses
       
       Returns
       -------
       I: obj:`numpy.ndarray`
           3x3 moment of inertia matrix
    """
    
    r = xyz - R_CM # Shift the position of the origin to the center of mass
    size = masses.shape[0]
    I = [[0,0,0],[0,0,0],[0,0,0]]
    for l in range(size):
        I[0][0] += masses[l] * (r[l][1]**2 + r[l][2]**2)
        I[0][1] += -masses[l] * r[l][1] * r[l][0]
        I[0][2] += -masses[l] * r[l][2] * r[l][0]
        I[1][1] += masses[l] * (r[l][0]**2 + r[l][2]**2)
        I[1][0] += -masses[l] * r[l][0] * r[l][1]
        I[1][2] += -masses[l] * r[l][2] * r[l][1]
        I[2][2] += masses[l] * (r[l][0]**2 + r[l][1]**2)
        I[2][0] += -masses[l] * r[l][0] * r[l][2]
        I[2][1] += -masses[l] * r[l][1] * r[l][2]
    return np.array(I)

def principal_axis(I, method):
    """Diagonalize the moment of inertia matrix
       and calculate the principal axis of inertia
       
       Parameters
       ----------
       I: obj:`numpy.ndarray`
           3x3 moment of inertia matrix
       method: obj:`str`
           method to use for diagonalization: 'mpmath' (recommended) or 'scipy'
       
       Returns
       -------
       I_prime: obj:`numpy.ndarray`
           1x3 principal axis of inertia
    """
    
    if method == 'mpmath':
        from mpmath import mp, eig, matrix # from https://mpmath.org/
        mp.prec = 128
        I_prime, X  = eig(matrix(I))
        return np.array(I_prime,dtype=np.float64)
    elif method == 'scipy':
        from scipy.linalg import eig
        I_prime, X = eig(I)
        return np.abs(I_prime)

def B(I_prime):
    """Calculate the Rotational Constant 
       using the Quantum Rigid Rotor Approximation
    
       Parameters
       ----------
       I_prime: obj:`numpy.ndarray`
           1x3 principal axis of inertia
       
       Returns
       -------
       Theta: obj:`numpy.ndarray`
           1x2 or 1x3 rotational constants
    """
    from scipy import constants
    hbar = constants.hbar
    if np.any(I_prime == 0):
        indexes = np.where(I_prime != 0)
        value = I_prime[indexes]
    else:
        value = I_prime
    Theta = (hbar**2) / (2 * value)
    return Theta 

def vib_degrees_freedom(atoms, Theta):
    """Vibrational degrees of freedom for molecule
    
       Parameters
       ----------
       atoms: obj:`numpy.ndarray`
           Atomic symbols
       Theta: obj:`numpy.ndarray`
           1x2 or 1x3 rotational constants
       
       Returns
       -------
       vib_df: obj:`numpy.int64`
           vibrational degrees of freedoms
    """
    trans_df = 3
    total_df = 3 * atoms.shape[0]
    rot_df = Theta.shape[0]
    vib_df = total_df - (trans_df + rot_df)
    return vib_df

def point_group_to_sigma(point_group):
    """Returns the rotational symmetry numbers 
       given the molecular point groups.
       
       Parameters
       ----------
       point_group: :obj:`str`
           the point group (using Gaussian16 notation)
       
       Returns
       -------
       sigma: :obj:`int`     
           the rotational symmetry number
       """
    if point_group in np.array(['C1','CI','CS','C*V']): # C_1, C_i, C_s, C_infty_v point groups
        sigma = 1
    elif point_group in np.array(['D*H']): # D_infty_h point group 
        sigma = 2
    elif point_group in np.array(['T','TD']): # T, T_d point groups
        sigma = 12
    elif point_group in np.array(['OH']): # O_h point group
        sigma = 24
    elif point_group in np.array(['IH']): # I_h point group
        sigma = 60
    elif point_group[0] in np.array(['S']): # S_n point group
        value = point_group.replace('S','')
        sigma = int(value)/2
    elif point_group[0] in np.array(['C']): # C_n, C_nh, C_nv point groups
        value = point_group.replace('C','').replace('H','').replace('V','')
        sigma = int(value)
    elif point_group[0] in np.array(['D']): # D_n, C_nh, D_nd point groups
        value = point_group.replace('D','').replace('H','')
        sigma = 2 * int(value)
    return sigma
    
def all_data(xyz, atoms, atom_molecule, point_group, method = 'mpmath'):
    """Returns total molecular mass, distance matrix, symmetry number, 
       vibrational degrees of freedoms, and rotational constants
       
       Parameters
       ----------
       xyz: obj:`numpy.ndarray`
           Standard orientation cartesian coordinates.
       atoms: obj:`numpy.ndarray`
           Atomic symbols
       atom_molecule: obj:`str`
           System is either an atom ('atom') or a molecule ('molecule')
       point_group: :obj:`str`
           the point group (using Gaussian16 notation)
       method: obj:`str`
           method to use for diagonalization: 'mpmath' (recommended) or 'scipy'
        
       
       Returns
       -------
       total_mass: obj:`numpy.float64`
           Sum of all atom masses (in g/mol)
       distance_matrix: obj:`numpy.ndarray`
           Calculated distance matrix (should agree with Gaussian16 output) in A.
       symmetry_numb: :obj:`int`   
           the rotational symmetry number
       vib_df: obj:`numpy.int64`
           vibrational degrees of freedoms
       rot_constants: obj:`numpy.ndarray`
           1x2 or 1x3 rotational constants (in si units)
       """
        
    total_mass, atom_masses = masses(atoms)
    if atom_molecule == 'atom':
        distance_matrix = 0
        symmetry_numb = 'None'
        vib_df = 0
        rot_constants = np.array([0, 0, 0], dtype=np.float64)
    elif atom_molecule == 'molecule':
        distance_matrix = distances(xyz)
        symmetry_numb = point_group_to_sigma(point_group)
        xyz_si, masses_si = convert(xyz, atom_masses)
        R_CM = center_mass(masses_si, xyz_si)
        I = moment_of_inertia(xyz_si, R_CM, masses_si)
        I_prime = principal_axis(I, method)
        rot_constants = B(I_prime)
        vib_df = vib_degrees_freedom(atoms, rot_constants)
    return (atom_masses, total_mass, distance_matrix, symmetry_numb, vib_df, rot_constants)