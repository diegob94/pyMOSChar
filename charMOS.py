import os
import os.path
import sys
import pickle
import spice3read
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess as sp
import shlex

class CharMOS:
    def __init__(self,
                modelFiles  = tuple(),
                libs        = dict(),
                mosLengths  = np.arange(1, 10, 1),
                simulator   = "ngspice",
                nmos        = "cmosn",
                pmos        = "cmosp",
                simOptions  = "",
                corners      = ('section=tt',),
                nmos_subckt_path  = None,
                pmos_subckt_path  = None,
                datFileName = "MOS.dat",
                vgsStep     =  25e-3,
                vdsStep     =  25e-3,
                vsbStep     =  25e-3,
                vgsMax      =  1.8,
                vdsMax      =  1.8,
                vsbMax      =  1.8,
                numfing     = 1,
                temp        = 300,
                width       = 1,
                scale       = 1e-6,
            ):

        for modelFile in modelFiles:
            if (not os.path.isfile(modelFile)):
                print("Model file {0} not found!".format(modelFile))
                print("Please call init() again with a valid model file")
                return None

        vgs = np.linspace(0, vgsMax, int(vgsMax/vgsStep + 1))
        vds = np.linspace(0, vdsMax, int(vdsMax/vdsStep + 1))
        vsb = np.linspace(0, vsbMax, int(vsbMax/vsbStep + 1))

        self.vgs = vgs
        self.vds = vds
        self.vsb = vsb
        self.simulator = simulator
        self.mosLengths = mosLengths
        self.modelFiles = modelFiles
        self.modelN = nmos
        self.modelP = pmos
        self.nmos_subckt_path = nmos_subckt_path
        self.pmos_subckt_path = pmos_subckt_path
        self.width = width
        self.vgsMax = vgsMax
        self.vgsStep = vgsStep
        self.vdsMax = vdsMax
        self.vdsStep = vdsStep
        self.simOptions = shlex.split(simOptions)
        self.output_dir = Path.cwd()/"work"
        self.libs = libs
        self.scale = scale

        if (self.simulator == "ngspice"):
            self.netlist_writer = NgspiceNetlistWriter(self)
            self.simOptions = ["-b"] + self.simOptions
        elif (self.simulator == "spectre"):
            self.netlist_writer = SpectreNetlistWriter(self)
        else:
            print("ERROR: Invalid/Unsupported simulator specified")
            sys.exit(0)

        self.mosDat = {}
        self.mosDat['pfet'] = {}
        self.mosDat['nfet'] = {}
        self.mosDat['modelFiles'] = modelFiles
        self.mosDat['simulator'] = simulator

        for fet in ["nfet","pfet"]:
            self.mosDat[fet]['corners'] = corners
            self.mosDat[fet]['temp'] = temp
            self.mosDat[fet]['length'] = mosLengths
            self.mosDat[fet]['width'] = width
            self.mosDat[fet]['numfing'] = numfing

            # 4D arrays to store MOS data---->f(L,               VSB,      VDS,      VGS      )
            self.mosDat[fet]['id']  = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs)))
            self.mosDat[fet]['vt']  = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs)))
            self.mosDat[fet]['gm']  = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs)))
            self.mosDat[fet]['gmb'] = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs)))
            self.mosDat[fet]['gds'] = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs)))
            self.mosDat[fet]['cgg'] = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs)))
            self.mosDat[fet]['cgs'] = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs)))
            self.mosDat[fet]['cgd'] = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs)))
            self.mosDat[fet]['cgb'] = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs)))
            self.mosDat[fet]['cdd'] = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs)))
            self.mosDat[fet]['css'] = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs)))

        self.mosDat['nfet']['vgs'] = vgs
        self.mosDat['nfet']['vds'] = vds
        self.mosDat['nfet']['vsb'] = -vsb

        self.mosDat['pfet']['vgs'] = -vgs
        self.mosDat['pfet']['vds'] = -vds
        self.mosDat['pfet']['vsb'] = vsb

    def genDB(self):
        progTotal = len(self.mosLengths)*len(self.vsb)
        progCurr  = 0
        print("Data generation in progress. Go have a coffee...")

        for idxL in range(len(self.mosLengths)):
            for idxVSB in range(len(self.vsb)):

                print("Info: Simulating for L={0}, VSB={1}".format(idxL, idxVSB))

                log_file = self.output_dir/f"{self.simulator}_{idxL}_{idxVSB}.log"
                netlists = self.netlist_writer.genNetlist(self.mosLengths[idxL], self.vsb[idxVSB])

                if (self.simulator == "ngspice"):

                    self.runSim(netlists["mos"].name,netlists["mos"].parent,log_file)
                    simDat = spice3read.read(netlists["mos_raw"])

                    self.mosDat['nfet']['id'][idxL][idxVSB]  = simDat['i(id)']
                    self.mosDat['nfet']['vt'][idxL][idxVSB]  = simDat['vt']
                    self.mosDat['nfet']['gm'][idxL][idxVSB]  = simDat['gm']
                    self.mosDat['nfet']['gmb'][idxL][idxVSB] = simDat['gmb']
                    self.mosDat['nfet']['gds'][idxL][idxVSB] = simDat['gds']
                    self.mosDat['nfet']['cgg'][idxL][idxVSB] = simDat['cgg']
                    self.mosDat['nfet']['cgs'][idxL][idxVSB] = simDat['cgs']
                    self.mosDat['nfet']['cgd'][idxL][idxVSB] = simDat['cgd']
                    self.mosDat['nfet']['cgb'][idxL][idxVSB] = simDat['cgb']
                    self.mosDat['nfet']['cdd'][idxL][idxVSB] = simDat['cdd']
                    self.mosDat['nfet']['css'][idxL][idxVSB] = simDat['css']

#                    self.mosDat['pfet']['id'][idxL][idxVSB]  = simDat['i(id)']
#                    self.mosDat['pfet']['vt'][idxL][idxVSB]  = simDat['vt']
#                    self.mosDat['pfet']['gm'][idxL][idxVSB]  = simDat['gm']
#                    self.mosDat['pfet']['gmb'][idxL][idxVSB] = simDat['gmb']
#                    self.mosDat['pfet']['gds'][idxL][idxVSB] = simDat['gds']
#                    self.mosDat['pfet']['cgg'][idxL][idxVSB] = simDat['cgg']
#                    self.mosDat['pfet']['cgs'][idxL][idxVSB] = simDat['cgs']
#                    self.mosDat['pfet']['cgd'][idxL][idxVSB] = simDat['cgd']
#                    self.mosDat['pfet']['cgb'][idxL][idxVSB] = simDat['cgb']
#                    self.mosDat['pfet']['cdd'][idxL][idxVSB] = simDat['cdd']
#                    self.mosDat['pfet']['css'][idxL][idxVSB] = simDat['css']

                elif (self.simulator == "spectre"):

                    self.runSim(netlists["mos"].name,netlists["mos"].parent, log_file)
                    simDat = spice3read.read(netlists["mos_raw"], 'spectre')

                    if (self.subcktPath == ""):
                        nmos = "mn"
                        pmos = "mp"
                    else:
                        nmos = "mn." + self.subcktPath
                        pmos = "mp." + self.subcktPath

                    self.mosDat['nfet']['id'][idxL][idxVSB]  = simDat['{0}:ids'.format(nmos)]
                    self.mosDat['nfet']['vt'][idxL][idxVSB]  = simDat['{0}:vth'.format(nmos)]
                    self.mosDat['nfet']['gm'][idxL][idxVSB]  = simDat['{0}:gm'.format(nmos)]
                    self.mosDat['nfet']['gmb'][idxL][idxVSB] = simDat['{0}:gmbs'.format(nmos)]
                    self.mosDat['nfet']['gds'][idxL][idxVSB] = simDat['{0}:gds'.format(nmos)]
                    self.mosDat['nfet']['cgg'][idxL][idxVSB] = simDat['{0}:cgg'.format(nmos)]
                    self.mosDat['nfet']['cgs'][idxL][idxVSB] = simDat['{0}:cgs'.format(nmos)]
                    self.mosDat['nfet']['cgd'][idxL][idxVSB] = simDat['{0}:cgd'.format(nmos)]
                    self.mosDat['nfet']['cgb'][idxL][idxVSB] = simDat['{0}:cgb'.format(nmos)]
                    self.mosDat['nfet']['cdd'][idxL][idxVSB] = simDat['{0}:cdd'.format(nmos)]
                    self.mosDat['nfet']['css'][idxL][idxVSB] = simDat['{0}:css'.format(nmos)]

                    self.mosDat['pfet']['id'][idxL][idxVSB]  = simDat['{0}:ids'.format(pmos)]
                    self.mosDat['pfet']['vt'][idxL][idxVSB]  = simDat['{0}:vth'.format(pmos)]
                    self.mosDat['pfet']['gm'][idxL][idxVSB]  = simDat['{0}:gm'.format(pmos)]
                    self.mosDat['pfet']['gmb'][idxL][idxVSB] = simDat['{0}:gmbs'.format(pmos)]
                    self.mosDat['pfet']['gds'][idxL][idxVSB] = simDat['{0}:gds'.format(pmos)]
                    self.mosDat['pfet']['cgg'][idxL][idxVSB] = simDat['{0}:cgg'.format(pmos)]
                    self.mosDat['pfet']['cgs'][idxL][idxVSB] = simDat['{0}:cgs'.format(pmos)]
                    self.mosDat['pfet']['cgd'][idxL][idxVSB] = simDat['{0}:cgd'.format(pmos)]
                    self.mosDat['pfet']['cgb'][idxL][idxVSB] = simDat['{0}:cgb'.format(pmos)]
                    self.mosDat['pfet']['cdd'][idxL][idxVSB] = simDat['{0}:cdd'.format(pmos)]
                    self.mosDat['pfet']['css'][idxL][idxVSB] = simDat['{0}:css'.format(pmos)]

                progCurr += 1
                progPercent = 100 * progCurr / progTotal
                print("progress:",progPercent)

        print()
        print("Data generated. Saving...")
        pickle.dump(mosDat, open(datFileName, "wb"), pickle.HIGHEST_PROTOCOL)
        print("Done! Data saved in " + datFileName)

    def runSim(self, fileName, cwd, log_file):
        log_file = Path(log_file)
        cmd = [self.simulator] + self.simOptions + [fileName]
        print(f"subprocess.run> {shlex.join(cmd)} (cwd={cwd}, log_file={log_file})")
        with log_file.open("w") as log:
            r = sp.run(cmd, stdout=log, stderr=log, cwd=cwd, check=True)
        return r

class AbstractNetlistWriter(ABC):
    def __init__(self, char_mos):
        self.output_dir = Path(char_mos.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.char_mos = char_mos
    @abstractmethod
    def genNetlist(self, L, VSB):
        pass

class NgspiceNetlistWriter(AbstractNetlistWriter):
    def __init__(self, char_mos):
        super().__init__(char_mos)
    def genNetlist(self, L, VSB):
        netlist_path = self.output_dir/f'charMOS_{L}_{VSB}.net'
        raw_path = self.output_dir/f"out_{L}_{VSB}.raw"
        raw_file = raw_path.name
        with netlist_path.open("w") as netlist:
            netlist.write("Characterize MOSFETs\n")
            netlist.write("\n")
            for modelFile in self.char_mos.modelFiles:
                netlist.write(f".include {modelFile}\n")
            for lib_name,lib_path in self.char_mos.libs.items():
                netlist.write(f".lib {lib_path} {lib_name}\n")
            netlist.write(".param length={0}\n".format(L))
            netlist.write(".param mosChar_sb={0}\n".format(VSB))
            netlist.write("\n")
            netlist.write("vds  nDrain 0 dc 0\n")
            netlist.write("vgs  nGate  0 dc 0\n")
            netlist.write("vbs_n  nBulk_n  0 dc {-mosChar_sb}\n")
            netlist.write("vbs_p  nBulk_p  0 dc mosChar_sb\n")
            netlist.write("\n")
            mos_vars = [
                ["mn",self.char_mos.nmos_subckt_path,"nBulk_n",self.char_mos.modelN],
                ["mp",self.char_mos.pmos_subckt_path,"nBulk_p",self.char_mos.modelP]]
            mos_vars = [[
                mos if subckt_path is None else "X" + mos,
                mos if subckt_path is None else "m.X" + mos + "." + subckt_path,
                bulk_node,
                model]
                    for mos,subckt_path,bulk_node,model in mos_vars]
            for mos,_,bulk_node,model in mos_vars:
                netlist.write(f"{mos} nDrain nGate 0 {bulk_node} {model} L={{length*{self.char_mos.scale}}} W={{{self.char_mos.width}*{self.char_mos.scale}}}\n")
            netlist.write("\n")
            netlist.write(".options dccap post brief accurate\n")
            netlist.write(".control\n")
            netlist.write(f"save all\n")
            for _,mos_path,_,_ in mos_vars:
                netlist.write(f"+ @{mos_path}[id] \n")
                netlist.write(f"+ @{mos_path}[vth]\n")
                netlist.write(f"+ @{mos_path}[gm]\n")
                netlist.write(f"+ @{mos_path}[gmbs] \n")
                netlist.write(f"+ @{mos_path}[gds] \n")
                netlist.write(f"+ @{mos_path}[cgg] \n")
                netlist.write(f"+ @{mos_path}[cgs] \n")
                netlist.write(f"+ @{mos_path}[cgd] \n")
                netlist.write(f"+ @{mos_path}[cdd] \n")
                netlist.write(f"+ @{mos_path}[cbs] \n")
            netlist.write("\n")
            netlist.write("dc vgs 0 {0} {1} vds 0 {2} {3}\n".format(self.char_mos.vgsMax, self.char_mos.vgsStep, self.char_mos.vdsMax, self.char_mos.vdsStep))
            netlist.write("\n")
            for mos,mos_path,_,_ in mos_vars:
                suffix = mos[-1]
                netlist.write(f"let id_{suffix}   = @{mos_path}[id]\n")
                netlist.write(f"let vt_{suffix}   = @{mos_path}[vth]\n")
                netlist.write(f"let gm_{suffix}   = @{mos_path}[gm]\n")
                netlist.write(f"let gmb_{suffix}  = @{mos_path}[gmbs]\n")
                netlist.write(f"let gds_{suffix}  = @{mos_path}[gds]\n")
                netlist.write(f"let cgg_{suffix}  = @{mos_path}[cgg]\n")
                netlist.write(f"let cgs_{suffix}  = -@{mos_path}[cgs]\n")
                netlist.write(f"let cgd_{suffix}  = -@{mos_path}[cgd]\n")
                netlist.write(f"let cgb_{suffix}  = @{mos_path}[cgg] - (-@{mos_path}[cgs])-(-@{mos_path}[cgd])\n")
                netlist.write(f"let cdd_{suffix}  = @{mos_path}[cdd]\n")
                netlist.write(f"let css_{suffix}  = -@{mos_path}[cgs]-@{mos_path}[cbs]\n")
                netlist.write("\n")
            save_vars = ['id', 'vt', 'gm', 'gmb', 'gds', 'cgg', 'cgs', 'cgd', 'cgb', 'cdd', 'css']
            netlist.write(f"write {raw_file} {' '.join([i + '_' + s for i in save_vars for s in ['n','p']])}\n")
            netlist.write("exit\n")
            netlist.write(".endc\n")
            netlist.write(".end\n")
        return dict(mos=netlist_path, mos_raw=raw_path)

class SpectreNetlistWriter(AbstractNetlistWriter):
    def __init__(self, char_mos):
        super().__init__(char_mos)
    def genNetlist(self, L, VSB):
        netlist_path = self.output_dir/f'charMOS_{L}_{VSB}.scs'
        raw_path = self.output_dir/f"charMOS_{L}_{VSB}.raw"
        raw_file = raw_path.name
        if (self.char_mos.subcktPath == ""):
            nmos = "mn"
            pmos = "mp"
        else:
            nmos = "mn." + self.char_mos.subcktPath
            pmos = "mp." + self.char_mos.subcktPath
        with netlist_path.open("w") as netlist:
            netlist.write('//charMOS.scs \n')
            for modelFile, corner in zip(self.char_mos.modelFiles, self.char_mos.corners):
                netlist.write('include  "{0}" {1}\n'.format(modelFile, corner))
            netlist.write("parameters length={0}\n".format(L))
            netlist.write("parameters mosChar_sb={0}\n".format(VSB))
            netlist.write('save {0}:ids {0}:vth {0}:igd {0}:igs {0}:gm {0}:gmbs {0}:gds {0}:cgg {0}:cgs {0}:cgd {0}:cgb {0}:cdd {0}:cdg {0}:css {0}:csg {0}:cjd {0}:cjs {1}:ids {1}:vth {1}:igd {1}:igs {1}:gm {1}:gmbs {1}:gds {1}:cgg {1}:cgs {1}:cgd {1}:cgb {1}:cdd {1}:cdg {1}:css {1}:csg {1}:cjd {1}:cjs\n'.format(nmos, pmos))
            netlist.write('parameters mosChar_gs=0 mosChar_ds=0 \n')
            netlist.write('vdsn     (vdn 0)         vsource dc=mosChar_ds  \n')
            netlist.write('vgsn     (vgn 0)         vsource dc=mosChar_gs  \n')
            netlist.write('vbsn     (vbn 0)         vsource dc=-mosChar_sb \n')
            netlist.write('vdsp     (vdp 0)         vsource dc=-mosChar_ds \n')
            netlist.write('vgsp     (vgp 0)         vsource dc=-mosChar_gs \n')
            netlist.write('vbsp     (vbp 0)         vsource dc=mosChar_sb  \n')
            netlist.write('\n')
            netlist.write('mn (vdn vgn 0 vbn) {0} l=length*1e-6 w={1}e-6 multi=1 nf={2} _ccoflag=1\n'.format(self.char_mos.modelN, self.char_mos.width, self.char_mos.numfing))
            netlist.write('mp (vdp vgp 0 vbp) {0} l=length*1e-6 w={1}e-6 multi=1 nf={2} _ccoflag=1\n'.format(self.char_mos.modelP, self.char_mos.width, self.char_mos.numfing))
            netlist.write('\n')
            netlist.write(f'options1 options gmin=1e-13 dc_pivot_check=yes reltol=1e-4 vabstol=1e-6 iabstol=1e-10 temp=27 tnom=27 rawfmt=nutbin rawfile="./{raw_file}" save=none\n')
            netlist.write('sweepvds sweep param=mosChar_ds start=0 stop={0} step={1} {{ \n'.format(self.char_mos.vdsMax, self.char_mos.vdsStep))
            netlist.write('sweepvgs dc param=mosChar_gs start=0 stop={0} step={1} \n'.format(self.char_mos.vgsMax, self.char_mos.vgsStep))
            netlist.write('}\n')
        return dict(mos=netlist_path, mos_raw = raw_path)


