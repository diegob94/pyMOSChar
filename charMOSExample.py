#!/usr/bin/env python
import sys

from pyMOSChar import CharMOS

import numpy as np

# Specify the name of the MOSFET model. Simple way to do so
# is to create a schematic in Virtuoso that contains both
# nmos and pmos transistors. Then generate the netlist in
# ADE. You'll then be able to view the netlist and see what
# the name of the model is.
nmos = "sky130_fd_pr__nfet_01v8"
pmos = "sky130_fd_pr__pfet_01v8"

# Specify the MOSFET width in microns.
width = 1

# Specify the MOSFET lengths you're interested
# in. The following code creates an array of
# values from 0.1 to 5.1 in steps of 0.1. Note
# that the arange() function omits the last value
# so if you call np.arange(0.1, 5.1, 0.1), the
# last value in the array will be 0.5.
# MOS lengths are in microns. Don't keep the
# step size too small. Fine steps will use a 
# LOT of RAM can cause the machine to hang!
#                     start, stop, step
mosLengths = np.arange(0.15, 100, 5)

## Example 2 for lenghs
#mosLengths = np.concatenate(
#np.arange(0.1, 1, 0.1),
#np.arange(1, 10, 0.5),
#np.arange(10, 100, 10))

vsbMax = 1
vsbStep = 20e-3

## test
mosLengths = np.linspace(0.15, 100, 10)
vsbN = 10
vsbMax = 1
vsbStep = vsbMax/(vsbN-1)

# Initialize the characterization process. Modify
# the values below as per your requirements. Ensure
# that the step values aren't too small. Otherwise
# your RAM will get used up.
char_mos = CharMOS(
    simulator='ngspice',
    mosLengths=mosLengths,
    libs={"tt":"/home/diegob/eda/src/skywater-pdk/libraries/sky130_fd_pr/latest/models/sky130.lib.spice"},
    nmos=nmos,
    pmos=pmos,
    nmos_subckt_path = "msky130_fd_pr__nfet_01v8",
    pmos_subckt_path = "msky130_fd_pr__pfet_01v8",
    simOptions="",
    corners=("",),
    datFileName="sky130_PDK_W{0}u.dat".format(width),
    vgsMax=1,
    vgsStep=20e-3,
    vdsMax=1,
    vdsStep=20e-3,
    vsbMax=vsbMax,
    vsbStep=vsbStep,
    numfing=1,
    temp=300,
    width=width,
    scale = 1,
    max_cores=4
)

# This function call finally generates the required database.
char_mos.gen_db()

