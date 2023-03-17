from __future__ import annotations

# TODO: fix this Abomination
try:
    from . import d2c
except ImportError:
    import d2c

from typing import Callable, List
import control
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass, field
import json
import dataclasses_json
from dataclasses_json import dataclass_json
import pathlib
import re
from multiprocessing import Pool
import os
from copy import copy
import pandas as pd
from scipy.interpolate import interp1d

from tkinter.filedialog import askdirectory



NORMALIZING= 'normalizing'
STANDARDIZING = 'standardizing'
STANDARD_DATA_FILE_NAME = 'DATA/SISO_ID_DATA_{}.csv'

# Data manip



def find_io_files_folder() -> pathlib.Path:
    """This functions opens a windows file explorer to select a folder.

    Returns
    -------
    pathlib.Path
        This is the path of the selected folder. This folder should or will
        contain files used or produced by the following functions in this repo.

    Raises
    ------
    ValueError
        Raises an error if no folder is selected, which returns an empty path string.
    """
    MAIN_FILE_FOLDER = askdirectory(title='Select Input/Output Files Folder')
    if MAIN_FILE_FOLDER == '':
        raise ValueError('Must select an Input/Output Files Folder')
    else:
        MAIN_FILE_FOLDER = MAIN_FILE_FOLDER + '/'

    return pathlib.Path(MAIN_FILE_FOLDER)

@dataclass
class SensorData:
    """ This class is used to store data collected from inputs (u) and outputs (y)
    of a plant. This data can later be used to identify a model of this plant,
    and to verify the validity of this model, using the id_plant and test_plant
    methods for the PlantDesignOutcome class.

    Attributes
    ----------
    N : int
        Number of datapoints contained in this class. The same number of datapoints
        should be found in t, r, u and y.
    T : float
        Average timestep between each timestamp in t.
    T_var : float
        Variation between all values of timesteps between timestamps in t.
    t : np.ndarray
        Timestamp at which each datapoint was collected. Should be in seconds.
    r : np.ndarray
        Command values at each timestamp. Since the system is identified in
        open-loop at the moment, the command value is not used. Corresponds to 
        the desired output value of the system/plant. The error betweent this
        desired output and the actual output is the signal that will be used to 
        drive a controller, which in turn drives the plant.
    u : np.ndarray
        Input values at each timestamp. Corresponds to the signal that drives the
        system/plant that should be identified.
    y : np.ndarray
        Output values at each timestamp. Corresponds to the actual output of the
        system/plant, as read by sensors and filters. The output collected here
        considred as the output of the plant when driven directly through the 
        input u in an open-loop fashion. 
    """
    N: int
    T: float
    T_var: float
    t: np.ndarray
    r: np.ndarray
    u: np.ndarray
    y: np.ndarray

    @classmethod
    def from_timeseries(cls, t:np.ndaraay, r: np.ndarray, u: np.ndarray,  y:np.ndarray) -> SensorData:
        t = t - t[0]
        N = t.shape[0]
        delta_t = t[1:]-t[:-1]
        T = np.mean(delta_t) # might not have very stable timestep, better to average
        T_var = np.var(delta_t)
        return cls(
            N, 
            T,
            T_var, 
            t, 
            r, 
            u, 
            y,
        )

    def plot(self):
        """Can be used to display the r, u and y timeseries contained in a 
        SensorData class.
        """
        fig, ax = plt.subplots(3, 1)
        ax[0].set_ylabel(r'$r(t)$')
        ax[1].set_ylabel(r'$u(t)$')
        ax[2].set_ylabel(r'$y(t)$')
        # Plot data
        ax[0].plot(self.t, self.r, label='Command', color='C0')
        ax[1].plot(self.t, self.u, label='Input', color='C1')
        ax[2].plot(self.t, self.y, label='Output', color='C2')
        for a in np.ravel(ax):
            a.set_xlabel(r'$t$ (s)')
            a.legend(loc='upper right')
        fig.tight_layout()
        
# Plant id-ing

# Generate plant training data and params before training

@dataclass
class PlantPreGeneratedDesignParams:
    """This class is used to specify parameters used by the PlantDesignOutcome 
    class to train a plant model using its id_plant method. A set of these
    parameters

    Attributes
    ----------
    num_order : int
        The numerator order of the plant model to be trained, which is 
        represented as a transfer fucntion.
    denum_order : int
        The denominator order of the plant model to be trained, which is 
        represented as a transfer fucntion.
    better_cond_method: str
        Describes how input (u) and output (y) data from the SensorData class
        will be manipulated in the id_plant method of the PlantDesignOutcome 
        class to better. Either allowed values for this attributes are defined
        by the NORMALIZING and STANDARDIZING global variables.

    """
    num_order                   : int = field(default=0 )
    denum_order                 : int = field(default=1 )
    better_cond_method          : str = field(default='')

def build_regularization_array(
    expBegin: int, 
    expEnd: int, 
    mantEnd: float, 
    maxNbrRegVals: int,
    ) -> np.ndarray:
    """Builds an array of regularization values. Regularization to help better conditioning when
    training a plant model usign least-squares.

    Parameters
    ----------
    expBegin : int
        the first value in the array will be a number x such that 
        x = 0 * 10^(expBegin)
    expEnd : int
        the last value in the array will be a number x such that 
        x = mantEnd * 10^(expEnd)
    mantEnd : float
        the last value in the array will be a number x such that 
        x = mantEnd * 10^(expEnd)
    maxNbrRegVals : int
        maximum value of values in the array. It is not guaranteed this number
        of values will be found in the array, since all zero values will be
        removed but one.

    Returns
    -------
    np.ndarray
        An array of values growing from zero to bigger and bigger values, with
        each value having a mantissa and an exponent in base ten growing bigger
        and bigger

    Raises
    ------
    ValueError
        Need to give this function a valid mantissa value in base 10.
    """
    if mantEnd <= 0.0 or mantEnd >= 10.0:
        raise ValueError("A mantissa in base 10 is by defintion between zero and 10.")

    nbrVals = int(np.sqrt(maxNbrRegVals))
    a1 = np.logspace(expBegin, expEnd, nbrVals)
    a2 = np.arange(0,mantEnd,mantEnd/nbrVals)
    regs = np.outer(a1, a2).flatten()
    regs.sort()

    regs = np.concatenate((np.array([0]), regs[ regs != 0 ]))

    # print(regs)
    return regs

@dataclass
class PlantDesignParams(PlantPreGeneratedDesignParams):
    """This class derived from PlantPreGeneratedDesignParams contains additional
    parameters used during plant model identification in the id_model method of
    the PlantDesignOutcome class.

    These paramaters describe a comple training scenario and are constructed,
    with very specific and/or global parameters added on top of the orginal
    parameters found in the PlantPreGeneratedDesignParams class.

    Attributes
    ----------
    regularization : float
        Training with more regularization will help to better condition the 
        least-squares problem and reduce overfitting. This paramater is very
        specific and is intended to differ in every training/testing scenario 
        produced when building a PlantDesignOutcome class and calling its
        id_plant and train_plant methods.
    sensor_data_column_names: 
        Associates the header of each column of a csv file used to create 
        instances of a SensorData class to a specific member of that class. Each
        column in the csv file is a np.ndarray in SensorData. This parameter is
        the most global one and should be identical for each instance of a 
        PlantDesignOutcome class, and the subsequent call to id_plant.
    """
    regularization:             float           = field(default=0)
    sensor_data_column_names:   dict[str, str]  = field(default_factory=dict)

    @classmethod
    def complete_design_params(
        cls, 
        pregen: PlantPreGeneratedDesignParams, 
        regularization: float = 0, 
        sensor_data_column_names: dict[str, str] = {}

    ) -> PlantDesignParams:
        """Used to complete a PlantPreGeneratedDesignParams instance with global 
        and specific parameters.

        Parameters
        ----------
        pregen : PlantPreGeneratedDesignParams
            Information used to train a plant model, usually found in a file
            with a list of such pregen'ed params.           
        regularization : float, optional
            This is a specific parameter. Can be used to penalize high values of
            discrete transfer function coefficients, by default 0
        sensor_data_column_names : dict[str, str], optional
            This is a global parameter. Used to identify which columns in a csv 
            file correspond to input and output signals of the plant, by default {}

        Returns
        -------
        PlantDesignParams
            Complete set of parameters used to specify how a plant should be trained,
            with information on the data used to do so and information on how that
            data should be processed before training.
        """
        return cls(
            pregen.num_order, 
            pregen.denum_order,
            pregen.better_cond_method,
            regularization,
            sensor_data_column_names
        )

@dataclass_json
@dataclass
class PlantDesignPlan:
    """This class is intended to be genereated from a *.json file containing a
    set of parameters, each set corresponding to one specific type of plant 
    training scenario. This file, and this class, also contain more generic
    parameters that should be applied to every training scenario. 
    Finally, every training scenario, in the form of 
    PlantPreGeneratedDesignParams instances in the plants_to_train attribute, 
    should be completed with the global paramaters in PlantDesignPlan and more
    specific parameters to produce even more training scenarios for the
    id_plant method to run in the PlantDesignOutcome class.

    Attributes
    ----------
    sensor_data_column_names : dict[str, str], optional
        This is a global parameter. Used to identify which columns in a csv 
        file correspond to input and output signals of the plant, by default {}
    plants_to_train : list[PlantPreGeneratedDesignParams]
        Every training scenario is describded by a PlantPreGeneratedDesignParams
        instance in this list. Many specific scenarios can be generated from
        these scenarios by adding a regularization value to the attributes 
        in a PlantPreGeneratedDesignParams instance and creating a 
        PlantDesignParams class.

    """
    sensor_data_column_names:   dict[str, str]                      = field(default_factory=dict)
    plants_to_train:            list[PlantPreGeneratedDesignParams] = field(default_factory=list)

    @classmethod
    def from_file(cls, file_path: pathlib.Path) -> PlantDesignPlan:
        """ Takes a *.json file withh all information necessary to build a 
        PlantDesignPlan class instance and creates such an instance.

        Parameters
        ----------
        file_path : pathlib.Path
            Where the *.json file is located in the file system of the computer 
            using this package.

        Returns
        -------
        PlantDesignPlan
            See class docstring above.
        """
        with open(file_path, 'r') as plant_json_file:
            plant_json_string = plant_json_file.read()

            plant_design_plan = cls.from_json(plant_json_string)

        return plant_design_plan

def tf_encoder(t: control.TransferFunction) -> str:
    """ This function is used to serialize an instance of the 
    control.TransferFunction class, in a way that is somewhat readable, and that
    can be a valid string to be inserted in a json file.

    Parameters
    ----------
    t : control.TransferFunction
        Transfer function to be serialized.

    Returns
    -------
    str
        Serialized, json valid, string
    """
    return t.__repr__()

def tf_decoder(s_t: str) -> control.TransferFunction:
    """This function is used to deserialize a string that contains all the
    coefficients in a control.TransferFucntion instance. This string has to have
    been generated using the tf_encoder function found in this module.

    Parameters
    ----------
    s_t : str
        String to deserialize. Usually originates from a *.json file.

    Returns
    -------
    control.TransferFunction
        Deserialized transfer function

    Raises
    ------
    TypeError
        Since transfer functions are serialized in a format such that they can 
        be interpreted as a list by a python interpreter, if the string input to 
        this function is not a string that can be interpreted as such, this 
        function will raise an error.
    """
    
    return eval(s_t.replace('TransferFunction', 'control.TransferFunction').replace('array', 'np.array'))

@dataclass_json
@dataclass
class Plant:
    delta_y_over_delta_u_c: control.TransferFunction = field(metadata=dataclasses_json.config(encoder=tf_encoder,decoder=tf_decoder))
    mean_u : float = field(default=0)
    mean_y : float = field(default=0)
    name : str = field(default="")
    discrt_coefs : List[int] = field(default_factory=list)

    @classmethod
    def from_file(cls, file_path: pathlib.Path) -> Plant:
        with open(file_path, 'r') as pl_file:
            pl_json = json.load(pl_file)

        return cls.from_dict(pl_json)

    def to_file(self, file_path: pathlib.Path):
        self_dict = self.to_dict()
        with open(file_path, 'w') as plant_file:
            plant_file.write(json.dumps(self_dict, indent=4))

@dataclass
class PlantTimeseriesStats:
    maximum   : float  = field(default=0, repr=False)
    mean      : float  = field(default=0, repr=False)
    std_dev   : float  = field(default=0, repr=False)

    def __init__(self, timeseries: np.ndarray, for_normalizing: bool):
        if for_normalizing:
            self.maximum = np.max(np.abs(timeseries))
        else:
            self.mean = np.mean(timeseries)
            self.std_dev = np.std(timeseries)

@dataclass
class PlantOrganizedSensorData:
    bcm     : str                   = field(default=''              , repr=False)
    reg     : float                 = field(default=0               , repr=False)


    N       : int                   = field(default=0               , repr=False)
    T       : float                 = field(default=0               , repr=False)
    A       : np.ndarray            = field(default=np.zeros([0])   , repr=False)
    b       : np.ndarray            = field(default=np.zeros([0])   , repr=False)
    n       : int                   = field(default=0               , repr=False)
    m       : int                   = field(default=0               , repr=False)
    u_stat  : PlantTimeseriesStats  = field(default=None            , repr=False)
    y_stat  : PlantTimeseriesStats  = field(default=None            , repr=False)
    ATA     : np.ndarray            = field(default=np.zeros([0])   , repr=False)
    
    # x is set by derived classes
    x       : np.ndarray            = field(default=np.zeros([0])   , repr=False)

    def __init__(self, s_data: SensorData, params: PlantDesignParams):
        self.bcm = params.better_cond_method
        if (not self.bcm == NORMALIZING and not self.bcm == STANDARDIZING) and self.bcm:
            raise ValueError('Must use either \'{NORMALIZING}\' or \'{STANDARDIZING}\' as method to improve data conditionning')
        self.reg = params.regularization

        self.N = s_data.N
        self.T = s_data.T
        u = s_data.u
        y = s_data.y

        # Form A and B matrices for system ID
        nbr_alp = params.denum_order
        nbr_bet = params.num_order + 1

        # Find how many datapoints need to be removed from length of u and y vector to set number of rows to A matrix
        lim = max(nbr_alp, nbr_bet)
        uk = u[::-1]
        yk = y[::-1]

        # We do this to imrpove conditionning
        if self.bcm == NORMALIZING:
            # Normalize Data
            self.u_stat = PlantTimeseriesStats(u, True)
            self.y_stat = PlantTimeseriesStats(y, True)
            uk = uk / self.u_stat.maximum
            yk = yk / self.y_stat.maximum
        
        if self.bcm == STANDARDIZING:
            # Standardize data
            self.u_stat = PlantTimeseriesStats(u, False)
            self.y_stat = PlantTimeseriesStats(y, False)

            uk = (uk - self.u_stat.mean)/self.u_stat.std_dev
            yk = (yk - self.y_stat.mean)/self.y_stat.std_dev

        self.b = yk[:-lim].reshape(-1, 1)
        self.A = np.zeros((self.N - lim, nbr_alp+nbr_bet))
        for i in range(nbr_alp):
            end_ind = 1-lim+i
            if end_ind == 0:
                self.A[:, [i]] = -yk[(i+1):].reshape(-1, 1)
            else:
                self.A[:, [i]] = -yk[(i+1):end_ind].reshape(-1, 1)
        for i in range(nbr_bet):
            end_ind = 1-lim+i
            if end_ind == 0:
                self.A[:, [nbr_alp+i]] = uk[(i+1):].reshape(-1, 1)
            else:
                self.A[:, [nbr_alp+i]] = uk[(i+1):end_ind].reshape(-1, 1)

        self.n = nbr_alp
        self.m = nbr_bet-1

        
        self.ATA = self.A.T @ self.A
        # print(ATA)

@dataclass
class PlantOrganizedIDProblem(PlantOrganizedSensorData):
    
    def __init__(self, s_data: SensorData, params: PlantDesignParams):
        PlantOrganizedSensorData.__init__(self, s_data, params)

    def train_x(self):
        size_I = self.A.shape[1]
        self.x =  np.linalg.solve(self.ATA + self.reg*self.reg*np.eye(size_I), self.A.T @ self.b)
        # print(f'x={x}')

    def find_c_plant_from_x(self) -> Plant:
        # Compute TF 
        # Extract denominator and numerator coefficents.
        Pd_ID_den = np.hstack([1, self.x[0:self.n, :].reshape(-1,)])  # denominator coefficents of DT TF
        Pd_ID_num = self.x[self.n:, :].reshape(-1,)  # numerator coefficents of DT TF

        # Compute DT TF (and remember to ``undo" the normalization or standardization).
        
        if self.bcm == NORMALIZING:
            # Normalizing
            Pd_ID = self.y_stat.maximum / self.u_stat.maximum * control.tf(Pd_ID_num, Pd_ID_den, self.T)
            Pc_ID = d2c.d2c(Pd_ID)
            plant = Plant(Pc_ID)

        if self.bcm == STANDARDIZING:
            # Standardizing:
            Pd_ID = self.y_stat.std_dev / self.u_stat.std_dev * control.tf(Pd_ID_num, Pd_ID_den, self.T)
            Pc_ID = d2c.d2c(Pd_ID)
            plant = Plant(Pc_ID,self.u_stat.mean,self.y_stat.mean)

        plant.discrt_coefs = self.x.flatten().tolist()
        return plant

@dataclass
class PlantTestingOrganizedSensorData:

    def __init__(self, s_data: SensorData, params: PlantDesignParams):
        PlantOrganizedSensorData.__init__(self, s_data, params)

    def set_x_for_testing(self, x:np.ndarray):
        self.x = x

@dataclass
class PlantGraphDataCommands:
    show        : bool = field(default=False)
    save_graph  : bool = field(default=False)
    save_data   : bool = field(default=False)
    index2graph : int  = field(default=0    )

    def pick_index2graph(self, training_dataset_index: int, qty_datasets: int):

        if training_dataset_index == qty_datasets:
            self.index2graph = 0
        else:
            self.index2graph = training_dataset_index-1

@dataclass
class PlantGraphData:
    t       : np.ndarray
    u       : np.ndarray
    y       : np.ndarray
    t_ID    : np.ndarray
    y_ID    : np.ndarray
    e       : np.ndarray
    e_rel   : np.ndarray

    def listen_to_cmds(self, cmds: PlantGraphDataCommands, name: str):
        if cmds.show or cmds.save_graph:
            self._show_data(cmds, name)
        
        if cmds.save_data:
            self._save_data(name)

    def _show_data(self, cmds: PlantGraphDataCommands, name: str):
        # Plot test data
        fig, ax = plt.subplots(2, 1)   
        ax[0].set_ylabel(r'$u(t)$ (Pa)')
        ax[1].set_ylabel(r'$y(t)$ (N)')
        # Plot data
        ax[0].plot(self.t, self.u, '--', label='input', color='C0')
        ax[1].plot(self.t, self.y, label='output', color='C1')
        ax[1].plot(self.t_ID, self.y_ID, '-.', label="IDed output", color='C2')
        for a in np.ravel(ax):
            a.set_xlabel(r'$t$ (s)')
            a.legend(loc='upper right')
        fig.suptitle(name)
        fig.tight_layout()

        path_name = 'plant_test_plots'
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        if cmds.save_graph:
            plt.savefig(f'{path_name}//{name}_io.png')
        # Plot error
        fig, ax = plt.subplots(2, 1)
        # Format axes
        for a in np.ravel(ax):
            a.set_xlabel(r'$t$ (s)')
        ax[0].set_ylabel(r'$e_{abs}(t)$ (N)')
        ax[1].set_ylabel(r'$e_{rel}(t) \times 100\%$ (unitless)')
        # Plot data
        ax[0].plot(self.t, self.e)
        ax[1].plot(self.t, self.e_rel)
        # for a in np.ravel(ax):
        #     a.legend(loc='lower right')
        fig.tight_layout()
        if cmds.save_graph:
            plt.savefig(f'plant_test_plots//{name}_e.png')
        # Show plots
        if cmds.show:
            plt.show()
    
    def _save_data(self, name: str):
        json_data_string = self.to_json()
        with open(f'{name}.json', 'w') as design_results_file:
            design_results_file.write(json_data_string)

@dataclass
class StatisfactionCriteria:
    conditioning        : bool = field(default=False)
    sigma               : bool = field(default=False)
    relative_uncertainty: bool = field(default=False)
    NMSE                : bool = field(default=False)

@dataclass
class PlantPerformance:
    performance_of          : str                   = field(default='training'                          )
    conditioning            : float                 = field(default=0.0                                 )
    sigma                   : np.ndarray            = field(default=np.zeros([0])           , repr=False, metadata=dataclasses_json.config(exclude=dataclasses_json.cfg.Exclude.ALWAYS))
    relative_uncertainty    : np.ndarray            = field(default=np.zeros([0])           , repr=False, metadata=dataclasses_json.config(exclude=dataclasses_json.cfg.Exclude.ALWAYS))
    MSE                     : float                 = field(default=0.0                     , repr=False)
    MSO                     : float                 = field(default=0.0, repr=False                     )
    NMSE                    : float                 = field(default=0.0                                 )

    is_completly_satisfying : bool                  = field(default=False                               )
    satisfying_attributes   : StatisfactionCriteria = field(default=None , repr=False)

    def _raise_error_impossible_scenario(self):
        raise ValueError(
            f"""Can only train, validate, or test a plant. It is not possible to evaluate the performance of {self.performance_of}. 
            Try to use either the strings 'training', 'validation' or 'testing' in the  PlantPerformance constructor.""")

    def compute_conditioning(self, p: PlantOrganizedSensorData) -> float:
        #_, S, _ = np.linalg.svd(p.A, full_matrices=True)
        #self.conditioning = np.max(S) / np.min(S)
        w, _ = np.linalg.eig(p.ATA)
        self.conditioning = np.sqrt(np.max(w) / np.min(w))
        return self.conditioning

    def compute_performance(self, p: PlantOrganizedSensorData):
        # Compute the uncertainty and relative uncertainty.
        # We flatten arrays here so numpy doesn't allocate array of shape (p.b.shape[0], p.b.shape[0])
        Ax_flat = np.reshape(p.A @ p.x, newshape=(p.A.shape[0],))
        b_flat = np.reshape(p.b, newshape=(p.b.shape[0],))
        bmAx = b_flat - Ax_flat
        bmAx_2norm_squared = np.linalg.norm(bmAx) ** 2
        self.sigma =  (1 / (p.N - (p.n + p.m + 1))) * bmAx_2norm_squared * np.linalg.inv((p.ATA))
        self.relative_uncertainty = np.divide(np.sqrt(np.diag(self.sigma).reshape(-1,1)), np.abs(p.x)) * 100

        # Compute the MSE, MSO, NMSE.
        self.MSE, self.MSO = 1/p.N*bmAx_2norm_squared, 1/p.N*np.linalg.norm(p.b) ** 2
        self.NMSE =  self.MSE/self.MSO    
        
    def determine_satisfaction(self):
        # Setup satisfactory limits 
        @dataclass 
        class StatisfactionCriteriaLimits:
            conditioning_low            : float
            conditioning_high           : float
            sigma_high                  : float
            relative_uncertainty_high   : float
            NMSE_high                   : float

        self.satisfying_attributes = StatisfactionCriteria()
        sat_crit_lim = StatisfactionCriteriaLimits(10, 700, 10, 10, 0.01)
        if self.performance_of == 'training':
            pass
        elif self.performance_of == 'validation':
            sat_crit_lim.conditioning_low-=3
            sat_crit_lim.conditioning_high+=100
            sat_crit_lim.sigma_high+=10
            sat_crit_lim.relative_uncertainty_high+=5
            sat_crit_lim.NMSE_high+=0.01
        elif self.performance_of == 'testing':
            sat_crit_lim.conditioning_low-=5
            sat_crit_lim.conditioning_high+=200
            sat_crit_lim.sigma_high+=20
            sat_crit_lim.relative_uncertainty_high+=10
            sat_crit_lim.NMSE_high+=0.02
        else:
            self._raise_error_impossible_scenario()
        
        # Verify we are within those limits
        if self.conditioning > sat_crit_lim.conditioning_low and self.conditioning < sat_crit_lim.conditioning_high:
            self.satisfying_attributes.conditioning = True
        if (self.sigma < sat_crit_lim.sigma_high).all():
            self.satisfying_attributes.sigma = True
        if (self.relative_uncertainty < sat_crit_lim.relative_uncertainty_high).all():
            self.satisfying_attributes.relative_uncertainty = True
        if self.NMSE < sat_crit_lim.NMSE_high:
            self.satisfying_attributes.NMSE = True

        if all(vars(self.satisfying_attributes).values()):
            self.is_completly_satisfying = True

@dataclass
class TestingStatisfactionCriteria:
    VAF: bool = field(default=False)
    FIT: bool = field(default=False)

@dataclass
class PlantTestingPerformance(PlantPerformance):
    performance_of                  : str                           = field(default='testing'                                   )
    # there is no conditioning that happens when testing or validating a plant, so we just set the conditioning to a satsifactory value
    conditioning                    : float                         = field(default=50                                          ) 
    VAF                             : float                         = field(default=0.0                                         )
    FIT                             : float                         = field(default=0.0                                         )

    testing_satisfying_attributes   : TestingStatisfactionCriteria  = field(default=None                            , repr=False)

    def _compute_conditioning(self, problem: PlantOrganizedIDProblem):
        raise TypeError("Cannot compute conditioning during testing or validation. If you wan to compute the conditioning of a matrix A computer using training data, please use the generic PlantPerformance constructor.")

    def compute_testing_performance(self, s_data_testing: SensorData, plant_from_training: Plant) -> PlantGraphData:
        if s_data_testing.T_var != 0.0: # this should only be possible if data if generated using np.arange or np.linspace
            t_end = s_data_testing.t[-1]
            
            intfu = interp1d(s_data_testing.t, s_data_testing.u)
            intfy = interp1d(s_data_testing.t, s_data_testing.y)

            s_data_testing.t = np.arange(0, t_end+s_data_testing.T, s_data_testing.T)
            try:
                s_data_testing.u = intfu(s_data_testing.t)
            except ValueError as e: 
                if str(e) == "A value in x_new is above the interpolation range.":
                    s_data_testing.t = s_data_testing.t[:-1]
                    s_data_testing.u = intfu(s_data_testing.t)
                else:
                    raise e
            s_data_testing.y = intfy(s_data_testing.t)

        delta_u = s_data_testing.u
        if plant_from_training.mean_u:
            delta_u = delta_u - plant_from_training.mean_u

        td_ID, delta_y = control.forced_response(
            plant_from_training.delta_y_over_delta_u_c, 
            s_data_testing.t, 
            delta_u
        )

        yd_ID = delta_y
        if plant_from_training.mean_y:
            yd_ID = yd_ID + plant_from_training.mean_y

        # Compute error
        e = yd_ID  - s_data_testing.y
        y_max = np.max(np.abs(s_data_testing.y))
        e_rel = np.abs(e) / y_max * 100 

        var_e = np.var(e)
        var_y_test = np.var(s_data_testing.y)
        self.VAF = (1 - var_e/var_y_test) * 100 
        #self.VAF = (1 - np.var(e)/np.var(s_data_testing.y)) * 100 
        
        RMSE =  np.sqrt(np.mean(e**2))
        std_y = np.sqrt(var_y_test)
        self.FIT = ( 1 - RMSE /  std_y) * 100

        return PlantGraphData(s_data_testing.t, s_data_testing.u, s_data_testing.y, td_ID, yd_ID, e, e_rel)

    def determine_satisfaction(self):
        # Check usual performance indicators first 
        PlantPerformance.determine_satisfaction(self)
        self.testing_satisfying_attributes = TestingStatisfactionCriteria()
        # Setup satisfactory limits for training specific performance indicators
        @dataclass 
        class TrainingStatisfactionCriteriaLimits:
            VAF_low: float
            FIT_low: float

        sat_crit_lim = TrainingStatisfactionCriteriaLimits(90,85)
        if self.performance_of == 'training':
            raise ValueError(f'PlantTestingPerformance cannot determine_satisfaction for the performance of training data. Please use the generic PlantPerformance instead')
        elif self.performance_of == 'validation':
            sat_crit_lim.VAF_low+=3
            sat_crit_lim.FIT_low+=3
        elif self.performance_of == 'testing':
            pass

        # Verify we are within those limits for training specific performance indicators
        if self.VAF > sat_crit_lim.VAF_low: 
            self.testing_satisfying_attributes.VAF = True
        if self.FIT > sat_crit_lim.FIT_low:
            self.testing_satisfying_attributes.FIT = True

        if all(vars(self.testing_satisfying_attributes).values()):
            self.is_completly_satisfying &= True
        else:
            self.is_completly_satisfying = False

@dataclass_json
@dataclass
class PlantDesignOutcome:
    name            : str                       = field(default=""  )
    params          : PlantDesignParams         = field(default=None)

    plant           : Plant                     = field(default=None)
   

    train_perform   : PlantPerformance          = field(default=None) 
    test_perform    : PlantTestingPerformance   = field(default=None)

    def id_plant(self, s_data: SensorData, params: PlantDesignParams):
        """
        Finds a plant transfer function (discrete and continous) based on
        input/output data, and computes how well this plant can recreate the
        training ouptut data from the training input data.

        Parameters
        ----------
        s_data : SensorData
            Training input/output data, with timestep in between data points and number of data points
        params : PlantDesignParams
            Specifies the plant transfer function shape (high-pass, low-pass, order, etc..)
        regularization : int, optional
            Can be used to penalize high values of discrete transfer function coefficients, by default 0

        Returns
        -------
        
        x: list
            Discrete plant coefficients

            
        """
        self.params = params
        # Create plant performance indicator
        self.train_perform = PlantPerformance('training')

        # Organise sensor data into computable characterization problem
        # This problem is discarded after training: we won't need to use training data again in the future
        problem = PlantOrganizedIDProblem(s_data, self.params)

        # See how well the characterisation problem is conditionned
        self.train_perform.compute_conditioning(problem)

        # Find the coefficients of the plant's discrete trasnfer function
        problem.train_x()

        # See how well we follow the output sensor data when putting the input data thru the plant
        self.train_perform.compute_performance(problem)

        # Look if plant performance is within acceptable bounds
        self.train_perform.determine_satisfaction()

        # Find the plant's continuous transfer function 
        self.plant = problem.find_c_plant_from_x()

        self.plant.name = self.name

    def test_plant(self, list_s_data_testing: List[SensorData], graph_data_cmds:PlantGraphDataCommands=None):
        # See how well the trained plant behaves when faced with fresh, unseen, input data
        list_int_test_perform = []
        for i, s_data_testing in enumerate(list_s_data_testing):
            int_test_perform = PlantTestingPerformance()
            org_testing_data = PlantTestingOrganizedSensorData(s_data_testing, self.params)

            org_testing_data.set_x_for_testing(np.array(self.plant.discrt_coefs))
            int_test_perform.compute_performance(org_testing_data)
            graph_data = int_test_perform.compute_testing_performance(s_data_testing, self.plant)

            if not graph_data_cmds:
                graph_data_cmds = PlantGraphDataCommands()

            if graph_data_cmds.index2graph == i:
                graph_data.listen_to_cmds(graph_data_cmds, self.name)

            list_int_test_perform.append(int_test_perform)

        self.average_int_testing_perform(list_int_test_perform)

        self.test_perform.determine_satisfaction()
        
        return graph_data

    def average_int_testing_perform(self, list_int_test_perform: List[PlantTestingPerformance]):
        self.test_perform = PlantTestingPerformance()

        self.test_perform.conditioning = 50
        
        list_sigma  = [p.sigma                  for p in list_int_test_perform]
        list_ru     = [p.relative_uncertainty   for p in list_int_test_perform]
        list_mse    = [p.MSE                    for p in list_int_test_perform]
        list_mso    = [p.MSO                    for p in list_int_test_perform]
        list_nmse   = [p.NMSE                   for p in list_int_test_perform]
        list_vaf    = [p.VAF                    for p in list_int_test_perform]
        list_fit    = [p.FIT                    for p in list_int_test_perform]

        self.test_perform.sigma                  = np.mean(list_sigma,   axis=0)
        self.test_perform.relative_uncertainty   = np.mean(list_ru,      axis=0)
        self.test_perform.MSE                    = np.mean(list_mse,     axis=0)
        self.test_perform.MSO                    = np.mean(list_mso,     axis=0)
        self.test_perform.NMSE                   = np.mean(list_nmse,    axis=0)
        self.test_perform.VAF                    = np.mean(list_vaf,     axis=0)
        self.test_perform.FIT                    = np.mean(list_fit,     axis=0)

@dataclass_json
@dataclass
class PlantDesignResult:
    outcomes: list[PlantDesignOutcome] = field(default_factory=list)

    def _make_parent_folder(self, file_path: pathlib.Path):
        if not file_path.parents[0].exists():
            file_path.parents[0].mkdir(parents=True)

    def save_results_to_file(self, file_path: pathlib.Path):
        # Put all trained plants in a file, with test results
        design_results_dict = self.to_dict()

        self._make_parent_folder(file_path)
        with open(file_path, 'w') as design_results_file:
            design_results_file.write(json.dumps(design_results_dict, indent=4))

    def save_plant_list_to_file(self, file_path: pathlib.Path):
        list_dict_plant = [o.plant.to_dict() for o in self.outcomes]

        self._make_parent_folder(file_path)
        with open(file_path, 'w') as design_plants_file:
            design_plants_file.write(json.dumps(list_dict_plant, indent=4))

@dataclass
class PlantProcessMaterial:
    name:                       str
    design_params:              PlantDesignParams
    graph_data_cmds:            PlantGraphDataCommands
    train_data_file_path:       pathlib.Path
    train_test_data_file_paths: List[pathlib.Path]

    def load_sensor_data(self, file_path: pathlib.Path) -> SensorData:
        df_raw = pd.read_csv(file_path, encoding = "utf-8")

        return SensorData.from_timeseries(
            t = np.array(df_raw[self.design_params.sensor_data_column_names['t']]),
            r = np.array(df_raw[self.design_params.sensor_data_column_names['r']]),
            u = np.array(df_raw[self.design_params.sensor_data_column_names['u']]),
            y = np.array(df_raw[self.design_params.sensor_data_column_names['y']]),
        )

        

    def load_train_test_data(self) -> tuple[SensorData, List[SensorData]]:
        train_sd = self.load_sensor_data(self.train_data_file_path)

        list_test_sd = []
        for file_path in self.train_test_data_file_paths:
            if file_path != self.train_data_file_path:
                list_test_sd.append(self.load_sensor_data(file_path))

        return train_sd, list_test_sd

class PlantProcessMaterialGenerator:
    train_test_data_folder:         pathlib.Path            = None
    graph_data_cmds:                PlantGraphDataCommands  = None
    plant_design_plan_file_name:    pathlib.Path            = None
    train_test_data_file_paths:     List[pathlib.Path]      = []
    plant_design_plan:              PlantDesignPlan         = None

    def __init__(
        self, 
        train_test_data_folder: pathlib.Path, 
        graph_data_cmds: PlantGraphDataCommands,
        plant_design_plan_file_name: pathlib.Path
    ):
        self.train_test_data_folder = train_test_data_folder
        self.graph_data_cmds = graph_data_cmds
        self.plant_design_plan_file_name = plant_design_plan_file_name
        # Process Material name constant sub strings
        self.pm_n_ss = (r'plant_n', r'_d', r'_', r'_ds', r'_reg')

        pattern = r'plant_n(?P<num>[0-9])_d(?P<den>[0-9])_(?P<bcm>st|no)_ds(?P<dfn>[0-9])_reg(?P<reg>[0-9]+\.[0-9]+)'
        self.p_re = re.compile(pattern)

        self._list_data_files()

        self.plant_design_plan = PlantDesignPlan.from_file(self.plant_design_plan_file_name)

    def give_process_material_list(self) -> List[PlantProcessMaterial]:
        raise NotImplementedError('This method should be implemeted by all derived classes, needs to provide a list of process_materials for Pool to generate processes.')

    def _list_data_files(self): 
        self.train_test_data_file_paths = sorted(self.train_test_data_folder.glob('*.csv'))
    
    @staticmethod
    def _find_p_m_name(params: PlantDesignParams, data_file_number: int) -> str:
        return f'plant_n{params.num_order}_d{params.denum_order}_{params.better_cond_method[0:2]}_ds{data_file_number}_reg{params.regularization}'

    def _p_m_from_name(self, name:str) -> PlantProcessMaterial:
        m = self.p_re.search(name)
        if m.group('bcm') == 'no':
            bcm = NORMALIZING
        elif m.group('bcm') == 'st':
            bcm = STANDARDIZING
        else:
            raise ValueError('Must use either \'{NORMALIZING}\' or \'{STANDARDIZING}\' as method to improve data conditionning')
        
        pregen_params = PlantPreGeneratedDesignParams(int(m.group('num')),int(m.group('den')),bcm)

        training_file_number = int(m.group('dfn'))

        # If we do this, we hope the user keeps the training files didn't change name
        # TODO make this cleaner, maybe append some type of unique id to each data file?
        train_data_file_path = self.train_test_data_file_paths[training_file_number]

        regularization = float(m.group('reg'))
        
        self.graph_data_cmds.pick_index2graph(training_file_number,len(self.train_test_data_file_paths))
        graph_data_cmds_i = copy(self.graph_data_cmds)
        
        design_params = PlantDesignParams.complete_design_params(
            pregen_params, 
            regularization=regularization,
            sensor_data_column_names=self.plant_design_plan.sensor_data_column_names
        )

        return PlantProcessMaterial(
            name, 
            design_params, 
            graph_data_cmds_i,
            train_data_file_path, 
            self.train_test_data_file_paths
        )

class PlantPMGTryAll(PlantProcessMaterialGenerator):
    
    reg_arr: np.ndarray = None
    def __init__(
        self, 
        train_test_data_folder: pathlib.Path,
        graph_data_cmds: PlantGraphDataCommands, 
        plant_design_plan_file_name : pathlib.Path, 
        regularization_array : np.ndarray
    ):
        super().__init__(
            train_test_data_folder, 
            graph_data_cmds,
            plant_design_plan_file_name
        )
       

        # Will break down the line if not in float, as int32 is not serializable 
        # by dataclass_json (bite me why)
        self.reg_arr = regularization_array.astype('float64')


    def give_process_material_list(self) -> List[PlantProcessMaterial]:
        # Find what shape the plants we wan't to train will have
        

        # Create list of all possible plant shapes, data files for train/testing,
        # regularization values combos

        process_material_list = []
        for pregen_params in self.plant_design_plan.plants_to_train:
            for i, train_data_file_path in enumerate(self.train_test_data_file_paths):
                self.graph_data_cmds.pick_index2graph(i, len(self.train_test_data_file_paths))
                graph_data_cmds_i = copy(self.graph_data_cmds)
                for regularization in self.reg_arr:
                #regularization = 0
                    design_params = PlantDesignParams.complete_design_params(
                        pregen_params, 
                        regularization=regularization,
                        sensor_data_column_names=self.plant_design_plan.sensor_data_column_names
                    )
                    name = self._find_p_m_name(design_params, i)
                    process_material = PlantProcessMaterial(
                        name, 
                        design_params, 
                        graph_data_cmds_i,
                        train_data_file_path, 
                        self.train_test_data_file_paths
                    )
                    process_material_list.append(process_material)

        return process_material_list

class PlantPMGTrySelection(PlantProcessMaterialGenerator):
    result_file_path: pathlib.Path = None
    def __init__(
        self, 
        train_test_data_folder: pathlib.Path,
        graph_data_cmds: PlantGraphDataCommands,
        plant_design_plan_file_name : pathlib.Path, 
        result_file_path: pathlib.Path
    ):
        super().__init__(
            train_test_data_folder, 
            graph_data_cmds,
            plant_design_plan_file_name
        )
        self.result_file_path = result_file_path

    def give_process_material_list(self) -> List[PlantProcessMaterial]:
        rnlist = []
        with open(self.result_file_path, 'r') as rnfile:
            rnlist = rnfile.readlines()

        process_material_list = []
        for rn in rnlist:
            rn = rn.rstrip()
            # skip to next line in file if empty line
            if rn == "":
                continue
            process_material = self._p_m_from_name(rn)

            process_material_list.append(process_material)

        return process_material_list

def test_all_possible_trained_plants(
    pmg : PlantProcessMaterialGenerator, 
    parallel_train_test_process : Callable[[PlantProcessMaterial], PlantDesignOutcome]
) -> PlantDesignResult:

    process_material_list = pmg.give_process_material_list()

    # Train and test plants using combos mentionned above
    with Pool() as pool:
        results = pool.imap(parallel_train_test_process, process_material_list)
        pool.close()
        pool.join()
    design_results = PlantDesignResult()
    design_results.outcomes = list(results)

    return design_results

def load_plant_list_from_file(file_path: pathlib.Path) -> List[Plant]:
    with open(file_path, 'r') as pl_tr_file:
        pl_tr_json = json.load(pl_tr_file)

    return [Plant.from_dict(d) for d in pl_tr_json]
