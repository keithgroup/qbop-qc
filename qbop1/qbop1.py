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

"""Main subroutine that does all calculation
   zpve_results :: compute the zpve energy in kcal/mol
   qbop_results :: compute the QBOP model 
   thermal_results :: compute the ideal gas rigid rotor harmonic oscillator thermal effects 
   total_thermal_S_Cv :: compute the total thermal effects, entropy, and C_v
   enthalpy_gibbs :: compute the enthalpy and Gibb's free energy
   """

import numpy as np

class QBOP: # qbop class
    
    def __init__(self, name, multiplicity, atom_molecule = 'molecule', 
                 zpebop_param_folder = 'opt_parameters/zpebop', 
                 qbop_param_folder = 'opt_parameters/qbop'):
        """Give the name of the ROHF output file to get the data from Gaussian output.
        
        Parameters
        ----------
        name: :obj:`str`
            name of the ROHF/CBSB3 output file or atom
        multiplicity: :obj:`numpy.int64`
            2S+1, where S is the number of unpaired spin
            states in a molecule
        atom_molecule: :obj:`str`,optional
            calculate properties of either an atom (atom_molecule = 'atom') 
            or molecule (atom_molecule = 'molecule')
        zpebop_param_folder: :obj:`str`,optional
            name of the path/folder containing zpe-bop2 parameters
        qbop_param_folder: :obj:`str`,optional
            name of the path/folder containing qbop-1 parameters
        """
        from . import read_output as ro
        from .spatial_geom import all_data
        
        self.atom_molecule = atom_molecule
        if zpebop_param_folder[-1] != '/':
            self.zpebop_param_folder = zpebop_param_folder + '/'
        else:
            self.zpebop_param_folder = zpebop_param_folder 
            
        if qbop_param_folder[-1] != '/':
            self.qbop_param_folder = qbop_param_folder + '/'
        else:
            self.qbop_param_folder = qbop_param_folder 
        
        if self.atom_molecule == 'molecule':
            self.mol, XYZ, self.point_group, self.mulliken = ro.read_entire_output(name) # get all data from the output file
        elif self.atom_molecule == 'atom':
            self.mol = np.array([name])
            XYZ = np.array([0,0,0])
            self.point_group = 'C*V'
            self.mulliken = 0
        self.masses, self.total_mass, self.distance_matrix, self.symmetry_numb, self.vib_df, self.rot_constants = all_data(XYZ, self.mol, self.atom_molecule, self.point_group)
        self.mult = multiplicity
        return None
    
    def zpve_results(self):
        """Compute zpve energy in kcal/mol using ZPE-BOP2
        
        Returns
        ------
        self.zpve: :obj:`numpy.float64`
            total zpve energy in kcal/mol
        self.E_net: :obj:`numpy.ndarray`
            net vibrational bond energy matrix
        """
        from . import zpebop2_equation as zpve_eq 
        if self.atom_molecule == 'molecule':
            self.zpe, two_bodies, three_two_decomp = zpve_eq.zpe(self.mol, self.mulliken, self.distance_matrix, self.zpebop_param_folder)
            self.E_net = two_bodies + three_two_decomp
        elif self.atom_molecule == 'atom':
            self.zpe = 0
            self.E_net = 0
        return (self.zpe, self.E_net)
    
    def qbop_results(self, T = 298.15, P = 1, units = 'kcal/mol'):
        """Compute vibrational partiton funtions 
           and thermal effects using bond orders 
           and populations contributions 
           
           Parameters
           ----------
           T: :obj:`numpy.float64`, optional
               temperature in K (default: 298.15 K)
           P: :obj:`numpy.float64`, optional
               pressure in atm (default: 1 atm)
           units: :obj:`str`, optional
               Units of energy (Default unit is kcal/mol). 
               There are two options: 
                 1. in Hartrees (units = 'Eh')
                 2. kcal/mol (units = 'kcal/mol')
               
           Returns
           -------
           qbop_info: :obj:`tuple`
               Contain vibrational partition function, thermal, 
               entropy, and specific capacity at constant volume contributions
           """
        from .qbop_equation import qbop
        
        self.T = T
        self.P = P
        if self.atom_molecule == 'molecule':
            E_v, q_v_bot, q_v_0, S_v, Cv_v = qbop(self.zpe, self.E_net, self.vib_df, self.mol, self.T, self.qbop_param_folder, self.mulliken)
        elif self.atom_molecule == 'atom':
            E_v = 0
            q_v_bot = 1
            q_v_0 = 1
            S_v = 0
            Cv_v = 0
        self.qbop_info = (E_v, q_v_bot, q_v_0, S_v, Cv_v)
        return self.qbop_info
    
    def thermal_results(self):
        """Calculate the ideal gas rigid-rotor harmonic oscillator approximation
        
        Returns
        -------
        self.data: :obj:`tuple`
            contains information on the individual electronic, translational, 
            rotational, and vibronic contributions for U, Cv, S, and q.  
        """
        from .rrho_model import thermal
        self.data = thermal(self.mol, self.total_mass, self.mult, self.rot_constants, self.symmetry_numb, self.T, self.P, self.qbop_info)
        return self.data
    
    def total_thermal_S_Cv(self, units = 'kcal/mol'):
        """Compute the total thermal, entropy, 
           and heat capacities at constant volume 
           contributions per the assigned unit
           
        Parameters
        ----------
        units: :obj:`str`, optional
            Units of energy (Default unit is kcal/mol). 
            There are two options: 
                1. in Hartrees (units = 'Eh')
                2. kcal/mol (units = 'kcal/mol')
        
        Returns
        -------
        self.thermal: :obj:`numpy.float64`
            total internal energy contributions per the assigned unit
        self.entropy: :obj:`numpy.float64`
            total entropy contributions per the assigned unit
        self.Cv: :obj:`numpy.float64`
            total heat capacity at constant volume per the assigned
            unit
        """
        self.units = units
        if self.units == 'Eh':
            conversion_factor = 627.5098
        elif self.units == 'kcal/mol':
            conversion_factor = 1
        
        thermal_contr, q, entropy_contr, Cv_contr = self.data
        
        self.thermal = np.sum(thermal_contr) * conversion_factor
        self.entropy = np.sum(entropy_contr) * conversion_factor
        self.Cv = np.sum(Cv_contr) * conversion_factor
        return (self.thermal, self.entropy, self.Cv)
    
    def enthalpy_gibbs(self):
        """Compute the total thermal enthalpy and Gibb's free energy 
           contributions using the ideal gas approximation
        
        Returns
        -------
        self.enthalpy: :obj:`numpy.float64`
            total enthalpy energy corrections
        self.gibbs: :obj:`numpy.float64`
            total Gibb's free energy corrections
        """
        k_B = 1.987204259e-3 # in kcal/mol*K
        if self.units == 'kcal/mol':
            k_B *= 1
        elif self.units == 'Eh':
            k_B *= 627.5098
        self.enthalpy = self.thermal + k_B * self.T
        self.gibbs = self.enthalpy - self.T * self.entropy
        return (self.enthalpy, self.gibbs)
