import os
from typing import Callable
import os.path
import sys

import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess as sp
import shlex
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses_json import DataClassJsonMixin
from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict
from threading import Lock
from scipy.interpolate import interpn

import pyMOSChar.spice3read as spice3read
from pyMOSChar.numpy_json import NumpyField

@dataclass
class FetData(DataClassJsonMixin):
    corners: List[str]
    temp: float
    width: float
    numfing: float
    length: np.ndarray = NumpyField()
    # 4D arrays to store MOS data---->f(L,VSB,VDS,VGS)
    id: np.ndarray = NumpyField()
    vt: np.ndarray = NumpyField()
    gm: np.ndarray = NumpyField()
    gmb: np.ndarray = NumpyField()
    gds: np.ndarray = NumpyField()
    cgg: np.ndarray = NumpyField()
    cgs: np.ndarray = NumpyField()
    cgd: np.ndarray = NumpyField()
    cgb: np.ndarray = NumpyField()
    cdd: np.ndarray = NumpyField()
    css: np.ndarray = NumpyField()
    # Pin voltages
    vgs: np.ndarray = NumpyField()
    vds: np.ndarray = NumpyField()
    vsb: np.ndarray = NumpyField()
    def __getitem__(self, key):
        return getattr(self,key)

@dataclass
class MosData(DataClassJsonMixin):
    pfet: FetData
    nfet: FetData
    modelFiles: List[str]
    libs: Dict[str,str]
    simulator: str
    @staticmethod
    def read_db(db_path):
        db_path = Path(db_path)
        return MosData.schema().loads(gzip.decompress(db_path.read_bytes()))
    def __getitem__(self, key):
        return getattr(self,key)
    def lookup(self, mosType, *outVars, **inVars):

        # Check if a valid MOSFET type is specified.
        mosType = mosType.lower()
        if (mosType not in ['nfet', 'pfet']):
            print("ERROR: Invalid MOSFET type. Valid types are 'nfet' and 'pfet'.")

        defaultL = min(self[mosType]['length'])
        defaultVGS = self[mosType]['vgs']
        defaultVDS = max(self[mosType]['vds'])/2;
        defaultVSB  = 0;

        # Figure out the mode of operation and the requested output arguments.
        # Mode 1 : Just one variable requested as output.
        # Mode 2 : A ratio or product of variables requested as output.
        # Mode 3 : Two ratios or products of variables requested as output.
        mode = 1
        outVarList = []

        if (len(outVars) == 2):
            mode = 3
            for outVar in outVars:
                if (type(outVar) == str):
                    if (outVar.find('/') != -1):
                        pos = outVar.find('/')
                        outVarList.append(outVar[:pos].lower())
                        outVarList.append(outVar[pos])
                        outVarList.append(outVar[pos+1:].lower())
                    elif (outVar.find('*') != -1):
                        pos = outVar.find('*')
                        outVarList.append(outVar[:pos].lower())
                        outVarList.append(outVar[pos])
                        outVarList.append(outVar[pos+1:].lower())
                    else:
                        print("ERROR: Outputs requested must be a ratio or product of variables")
                        return None
                else:
                    print("ERROR: Output variables must be strings!")
                    return None
        elif (len(outVars) == 1):
            outVar = outVars[0]
            if (type(outVar) == str):
                if (outVar.find('/') == -1 and outVar.find('*') == -1):
                    mode = 1
                    outVarList.append( outVar.lower())
                else:
                    mode = 2
                    if (outVar.find('/') != -1):
                        pos = outVar.find('/')
                        outVarList.append(outVar[:pos].lower())
                        outVarList.append(outVar[pos])
                        outVarList.append(outVar[pos+1:].lower())
                    elif (outVar.find('*') != -1):
                        pos = outVar.find('*')
                        outVarList.append(outVar[:pos].lower())
                        outVarList.append(outVar[pos])
                        outVarList.append(outVar[pos+1:].lower())
            else:
                print("ERROR: Output variables must be strings!")
                return None
        else:
            print("ERROR: No output variables specified")
            return None

        # Figure out the input arguments. Set to default those not specified.
        varNames = [key for key in inVars.keys()]

        for varName in varNames:
            if (not varName.islower()):
                print("ERROR: Keyword args must be lower case. Allowed arguments: l, vgs, cds and vsb.")
                return None
            if (varName not in ['l', 'vgs', 'vds', 'vsb']):
                print("ERROR: Invalid keyword arg(s). Allowed arguments: l, vgs, cds and vsb.")
                return None

        L = defaultL
        VGS = defaultVGS
        VDS = defaultVDS
        VSB = defaultVSB
        if ('l' in varNames):
            L = inVars['l']
        if ('vgs' in varNames):
            VGS = inVars['vgs']
        if ('vds' in varNames):
            VDS = inVars['vds']
        if ('vsb' in varNames):
            VSB = inVars['vsb']

        xdata = None
        ydata = None

        # Extract the data that was requested
        if (mode == 1):
            ydata = self[mosType][outVarList[0]]
        elif (mode == 2 or mode == 3):
            ydata = eval("self[mosType][outVarList[0]]" + outVarList[1] + "self[mosType][outVarList[2]]")
            if (mode == 3):
                xdata = eval("self[mosType][outVarList[3]]" + outVarList[4] + "self[mosType][outVarList[5]]")

        # Interpolate for the input variables provided
        if (mosType == 'nfet'):
            points = (self[mosType]['length'], -self[mosType]['vsb'], self[mosType]['vds'], self[mosType]['vgs'])
        else:
            points = (self[mosType]['length'],  self[mosType]['vsb'], -self[mosType]['vds'], -self[mosType]['vgs'])

        xi_mesh = np.array(np.meshgrid(L, VSB, VDS, VGS))
        xi = np.rollaxis(xi_mesh, 0, 5)
        xi = xi.reshape(int(xi_mesh.size/4), 4)

        len_L = len(L) if type(L) == np.ndarray or type(L) == list else 1
        len_VGS = len(VGS) if type(VGS) == np.ndarray or type(VGS) == list else 1
        len_VDS = len(VDS) if type(VDS) == np.ndarray or type(VDS) == list else 1
        len_VSB = len(VSB) if type(VSB) == np.ndarray or type(VSB) == list else 1

        if (mode == 1 or mode == 2):
            result = np.squeeze(interpn(points, ydata, xi).reshape(len_L, len_VSB, len_VDS, len_VGS))
        elif (mode == 3):
            print("ERROR: Mode 3 not supported yet :-(")

        # Return the result
        return result

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
                datFileName = "MOS",
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
                max_cores   = 1,
            ):

        self.mutex = Lock()

        for modelFile in modelFiles:
            if (not os.path.isfile(modelFile)):
                raise FileNotFoundError("Model file {0} not found! Please call init() again with a valid model file".format(modelFile))

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
        self.max_cores = max_cores
        self.run_simulations = True

        self.datFileName = Path(datFileName)
        if not self.datFileName.is_absolute():
            self.datFileName = self.output_dir / self.datFileName
        self.datFileName = self.datFileName.with_suffix('.json.gz')

        if (self.simulator == "ngspice"):
            self.netlist_writer = NgspiceNetlistWriter(self)
            self.simOptions = ["-b"] + self.simOptions
        elif (self.simulator == "spectre"):
            self.netlist_writer = SpectreNetlistWriter(self)
        else:
            print("ERROR: Invalid/Unsupported simulator specified")
            sys.exit(0)

        self.mosDat = MosData(
            modelFiles=modelFiles,
            libs=libs,
            simulator=simulator,
            nfet = FetData(
                corners=corners,
                temp=temp,
                length=mosLengths,
                width=width,
                numfing=numfing,
                id  = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                vt  = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                gm  = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                gmb = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                gds = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                cgg = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                cgs = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                cgd = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                cgb = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                cdd = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                css = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                vgs = vgs,
                vds = vds,
                vsb = -vsb,
            ),
            pfet = FetData(
                corners=corners,
                temp=temp,
                length=mosLengths,
                width=width,
                numfing=numfing,
                id  = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                vt  = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                gm  = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                gmb = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                gds = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                cgg = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                cgs = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                cgd = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                cgb = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                cdd = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                css = np.zeros((len(mosLengths), len(vsb), len(vds), len(vgs))),
                vgs = -vgs,
                vds = -vds,
                vsb = vsb,
            ),
        )
    def gen_db(self):
        print("Starting simulation sweep:")
        print(f"  Length: min={self.mosLengths.min()} max={self.mosLengths.max()} length={len(self.mosLengths)}")
        print(f"  VSB: min={self.vsb.min()} max={self.vsb.max()} length={len(self.vsb)}")

        # Generate simulation jobs
        jobs = []
        for idxL in range(len(self.mosLengths)):
            for idxVSB in range(len(self.vsb)):
                jobs.append(Job(
                    id=len(jobs),
                    params=dict(idxL=idxL,idxVSB=idxVSB),
                    function=self.run_job
                ))
        print(f"Executing {len(jobs)} simulation jobs with {self.max_cores} parallel cores")

        # Execute simulations
        with ThreadPoolExecutor(max_workers = self.max_cores) as executor:
            futures = [executor.submit(lambda x: x(), job) for job in jobs]
            for counter,future in enumerate(as_completed(futures)):
                try:
                    future.result()
                except Exception as e:
                    for f in futures:
                        f.cancel()
                    raise e from None
                print(f"Progress {100*(counter+1)/len(jobs):.2f}%")

        # Collect result data
        for job in jobs:
            self.netlist_writer.read_data(job.results["raw"],idxL=job.params['idxL'],idxVSB=job.params['idxVSB'])

        print()
        print("Data generated. Saving...")
        self.datFileName.write_bytes(gzip.compress(self.mosDat.to_json().encode()))
        print(f"Done! Data saved in {self.datFileName.resolve()}")

        return self.mosDat

    def run_job(self,idxL,idxVSB):
        self.mutex.acquire()
        print("Simulating for L={0}, VSB={1}".format(self.mosLengths[idxL], self.vsb[idxVSB]))
        log_file = self.output_dir/f"{self.simulator}_{idxL}_{idxVSB}.log"
        netlists = self.netlist_writer.genNetlist(self.mosLengths[idxL], self.vsb[idxVSB])
        cmd = [self.simulator] + self.simOptions + [netlists["mos"].name]
        if self.run_simulations:
            run_command(cmd,netlists["mos"].parent,log_file,
                before_run=lambda: self.mutex.release(),
                after_run=lambda: print(f"Simulating for L={self.mosLengths[idxL]}, VSB={self.vsb[idxVSB]} done"))
        else:
            self.mutex.release()
        return dict(raw=netlists['mos_raw'],log=log_file)

def run_command(cmd, cwd=Path.cwd(), log_file=None, before_run=lambda: None, after_run=lambda: None):
    run_args = {}
    if isinstance(cmd,str):
        cmd = shlex.split(cmd)
    if log_file is not None:
        log_file = Path(log_file)
        print(f"subprocess.run> {shlex.join(cmd)} (cwd={cwd}, log_file={log_file})")
        log = log_file.open('w')
        run_args = dict(stdout=log, stderr=log)
    before_run()
    r = sp.run(cmd, cwd=cwd, check=True, **run_args)
    after_run()
    if log_file is not None:
        log.close()
    return r

@dataclass
class Job:
    id: int
    function: Callable
    params: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    def __call__(self):
        self.results = self.function(**self.params)

class NetlistWriter(ABC):
    def __init__(self, char_mos):
        self.output_dir = Path(char_mos.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.char_mos = char_mos
    def get_tag(self,*args):
        return '_'.join([f'{i:.2f}'.replace('.','p') for i in args])
    @abstractmethod
    def genNetlist(self, L, VSB):
        pass
    @abstractmethod
    def read_data(self, raw_path, **kwargs):
        pass

class NgspiceNetlistWriter(NetlistWriter):
    def __init__(self, char_mos):
        super().__init__(char_mos)
    def genNetlist(self, L, VSB):
        netlist_path = self.output_dir/f'charMOS_{self.get_tag(L,VSB)}.net'
        raw_path = self.output_dir/f"out_{self.get_tag(L,VSB)}.raw"
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
            netlist.write('quit\n')
            netlist.write(".endc\n")
            netlist.write(".end\n")
        return dict(mos=netlist_path, mos_raw=raw_path)
    def read_data(self, raw_path, **kwargs):
        idxL = kwargs['idxL']
        idxVSB = kwargs['idxVSB']
        simDat = spice3read.read(raw_path)
        for c in ['n','p']:
            fet = getattr(self.char_mos.mosDat, c + 'fet')
            suffix = '_' + c
            fet.id[idxL][idxVSB]  = simDat[f'i(id{suffix})']
            fet.vt[idxL][idxVSB]  = simDat[f'v(vt{suffix})']
            fet.gm[idxL][idxVSB]  = simDat['gm'+suffix]
            fet.gmb[idxL][idxVSB] = simDat['gmb'+suffix]
            fet.gds[idxL][idxVSB] = simDat['gds'+suffix]
            fet.cgg[idxL][idxVSB] = simDat['cgg'+suffix]
            fet.cgs[idxL][idxVSB] = simDat['cgs'+suffix]
            fet.cgd[idxL][idxVSB] = simDat['cgd'+suffix]
            fet.cgb[idxL][idxVSB] = simDat['cgb'+suffix]
            fet.cdd[idxL][idxVSB] = simDat['cdd'+suffix]
            fet.css[idxL][idxVSB] = simDat['css'+suffix]

class SpectreNetlistWriter(NetlistWriter):
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
    def read_data(self, raw_path, **kwargs):
        idxL = kwargs['idxL']
        idxVSB = kwargs['idxVSB']
        simDat = spice3read.read(raw_path, 'spectre')

        if (self.char_mos.subcktPath == ""):
            nmos = "mn"
            pmos = "mp"
        else:
            nmos = "mn." + self.char_mos.subcktPath
            pmos = "mp." + self.char_mos.subcktPath

        for c,mos in [['n',nmos],['p',pmos]]:
            fet = c + 'fet'
            self.char_mos.mosDat[fet]['id'][idxL][idxVSB]  = simDat['{0}:ids'.format(mos)]
            self.char_mos.mosDat[fet]['vt'][idxL][idxVSB]  = simDat['{0}:vth'.format(mos)]
            self.char_mos.mosDat[fet]['gm'][idxL][idxVSB]  = simDat['{0}:gm'.format(mos)]
            self.char_mos.mosDat[fet]['gmb'][idxL][idxVSB] = simDat['{0}:gmbs'.format(mos)]
            self.char_mos.mosDat[fet]['gds'][idxL][idxVSB] = simDat['{0}:gds'.format(mos)]
            self.char_mos.mosDat[fet]['cgg'][idxL][idxVSB] = simDat['{0}:cgg'.format(mos)]
            self.char_mos.mosDat[fet]['cgs'][idxL][idxVSB] = simDat['{0}:cgs'.format(mos)]
            self.char_mos.mosDat[fet]['cgd'][idxL][idxVSB] = simDat['{0}:cgd'.format(mos)]
            self.char_mos.mosDat[fet]['cgb'][idxL][idxVSB] = simDat['{0}:cgb'.format(mos)]
            self.char_mos.mosDat[fet]['cdd'][idxL][idxVSB] = simDat['{0}:cdd'.format(mos)]
            self.char_mos.mosDat[fet]['css'][idxL][idxVSB] = simDat['{0}:css'.format(mos)]

