"""
This script opens a viewer window for the simulation in the directory 'example_path_to_simu_directory'."""
import sys

sys.path.extend([r'C:\Users\chenjie123456\Desktop\dark_matter\galaximview-master\galaximview'])

from galaximview import snap, galviewer
from matplotlib import interactive
interactive(True)

open_colliding_galaxies_sfr = True

if open_colliding_galaxies_sfr:
    example_path_to_simu_directory = r'C:\Users\chenjie123456\Desktop\dark_matter\output'
    simu = snap.IterOnSim(example_path_to_simu_directory, input_type='G4_hdf5')
else:
    example_path_to_simu_directory = r'C:\Users\chenjie123456\Desktop\dark_matter\G2-galaxy-G4run-G1G2format'
    simu = snap.IterOnSim(example_path_to_simu_directory, input_type='G1/G2', ntypes=3)

resize_text_if_too_large = False  # if the etxt is too large for the buttons

if resize_text_if_too_large:
    import matplotlib
    matplotlib.rcParams.update({'font.size': 8})

gv = galviewer.GalViewer(simu, figname=simu.direc, adaptive_zoom_ini=False, rmax=160)  # loads the viewer
buts = gv.create_buttons()  # this reference to the buttons must be kept for the buttons to work
gv.fig.subplots_adjust(bottom=0.12, top=0.9, left=0.001, right=0.93, wspace=0.12, hspace=0)