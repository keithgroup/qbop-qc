# QBOP

The vibrational partition function from from bond orders and populations (QBOP) program is a computational chemistry algorithm that computes vibrational  partition functions along with the thermal effects computed using the ideal gas rigid-rotor harmonic oscillator approximations.  These methods require Hartree-Fock orbital populations and bond orders from approximate quantum chemistry methods. Moreover, these methods do not require Hessians or higher-order derivatives. 

## Installation

```bash
git clone https://github.com/keithgroup/qbop-qc
cd qbop-qc
pip install .
```

## Preparing QBOP Input Files

Currently, we have developed QBOP-1 models, and each of these models require output from the MinPop algorithm. Below is an example of how to do this in Gaussian 16 for each model.

1. Optimize your molecular structure using your preferred level of theory (e.g., B3LYP/CBSB7 and B3LYP/cc-pVTZ+1d).

##### QBOP-1:

    # Opt B3LYP/cc-pVTZ+1d

2. Run Hartree-Fock or B3LYP on the optimized structure in Gaussian.

##### QBOP-1:

    # SP ROHF/CBSB3 Pop=(Full) IOp(6/27=122,6/12=3)

## Usage

To execute `qbop1`,  run this in the command line:

```bash
qbop1 -f {name_file} -mult {multiplicity} -T {temperature} -P {pressure} -param_folder {param_folder} --json > {name_file}.bop
```

where `{name_file}` is the Hartree-Fock MinPop output (or the atomic symbol of an atom) file, `{multiplicity}` is the multiplicity, `{temperature}` is the temperature in K, `{pressure}` is the pressure in atm, and `{param_folder}` is the name of the folder containing the QBOP-1 and ZPEBOP-2 parameter folders. Note that both models' parameters are stored in json files under the `opt_parameters` folders.

Some examples of QBOP-1 output files are found in the `examples` directory.

Some details of the parsers used in `qbop1` source codes.

##### QBOP-1:

```bash
$ qbop1.py -h
usage: qbop1.py [-h] -f F -mult MULT [-type TYPE] [-T T] [-P P] [-param_folder PARAM_FOLDER] [--json]

compute thermal energies using QBOP-1 and the ideal gas rigid-rotor harmonic oscillator method

optional arguments:
  -h, --help            show this help message and exit
  -f F                  name of the Gaussian Hartree-Fock output file or the atom symbol
  -mult MULT            multiplicity (i.e., 2S + 1)
  -type TYPE            is the file a molecule or atom (type: atom) (default: molecule)?
  -T T                  temperature in K (default:298.15 K)
  -P P                  pressure of the system (default:1 atm)
  -param_folder PARAM_FOLDER
                        name of ZPEBOP-2 and QBOP-1's parameter folder (default: opt_parameters/)
  --json                save the job output into JSON
```
Note the user can calculate the thermal effects from the first row atoms. Instead of following steps 1-2 chronologically, the user can insert the atom symbol under  `{name_file}` and change `{type}` from `molecule` to `atom`. 

## Citations

Please cite:

**QBOP-1**:  Barbaro Zulueta and John A. Keith. Vibrational Partition Functions from Bond Orders and Populations. (under preparation), 2025.

## License

Distributed under the MIT License.
See `LICENSE` for more information.
