import pyGDM2.core as code
from dataclasses import fields
from email.iterators import _structure
import numpy as np
import matplotlib.pyplot as plt
import tools
from ase.io import read


from pyGDM2 import fields
from pyGDM2 import propagators

from pyGDM2 import visu
from pyGDM2 import tools
from pyGDM2 import linear

from Epsilon_ExpClass import Ag

class NearFieldSimulator:
    def __init__(self, material, step_size=2.88, structure_file=None):
        """Initialize with nanostructure parameters"""
        if structure_file is not None:
                self.structure = read(structure_file)
                self.structure.positions = self.structure.get_center_of_mass()
                self.g = self.structure.positions
        else:
                self.structure = None
                self.g = None
                
        self.material = material
        self.step_size = step_size
        self.sim = None
        
    def setup_simulation(self, wavelength, polarization='linear'):
        """Configure simulation at specific wavelength"""
    
        if polarization == 'circular':
            self.field_kwargs = {'inc_angle': 180, 'E_s': 1, 'E_p': 1, 'phase_Es': [-np.pi/2]}
        else:  # linear
            self.field_kwargs = {'inc_angle': 0, 'E_s': 0, 'E_p': 1}
      
        field_gen = fields.plane_wave
        self.efield = fields.OffField(field_gen, 
                                  [wavelength],userkwargs=self.field_kwargs)
        self.dyads = propagators.DyadsQuasistatic(n_environment=1.0)
        self.struct = _structure.struct(self.step_size, self.g, [self.material()]*len(self.g))
        self.sim = code.simulation(self.structure, self.efield, self.dyads)
        self.sim.scatter(verbose=False)
        
    def calculate_nearfield(self, padding=2.0, resolution=101):
        """Calculate near-field around structure"""
        # Create probe grid (extending beyond structure)
        x_range = padding * max(abs(self.g[:,0]))
        y_range = padding * max(abs(self.g[:,1]))
        z_height = max(self.g[:,2]) + 2*self.step_size
        
        r_probe = tools.generate_NF_map(2*min(self.g[:,0]),2*max(self.g[:,0]),101, 
                                        2*min(self.g[:,1]),2*max(self.g[:,1]),101, 
                                        Z0=self.g.T[2].max()+2*self.Step)
        Es, Et, Bs, Bt = linear.nearfield(self.sim, 0, r_probe)
    
    def plot_enhancement(self, save_path=None):
        """Visualize field enhancement"""
        plt.figure(figsize=(10,8))
        
        # Plot structure
        visu.structure(self.g, show=False, alpha=0.3)
        
        # Plot field enhancement
        im = visu.vectorfield_color(self.Es, 
                                  tit=f'Field Enhancement at λ = {self.efield.wavelength[0]:.0f} nm',
                                  show=False)
        
        # Formatting
        plt.colorbar(im, label='$|E/E_0|^2$')
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Example Usage --------------------------------------------------
if __name__ == "__main__":
    # 1. Initialize with your structure
    simulator = NearFieldSimulator(
        structure_file="Structures/Ag_Ih_2057.xyz",
        material=Ag,  # From Epsilon_ExpClass
        step_size=2.88
    )
    
    # 2. Set your desired wavelength (in nm)
    wavelength = 520  # Example for silver nanostructures
    
    # 3. Run simulation and visualize
    simulator.setup_simulation(wavelength, polarization='circular')
    simulator.calculate_nearfield(padding=1.5, resolution=151)
    simulator.plot_enhancement(save_path='enhancement.png')