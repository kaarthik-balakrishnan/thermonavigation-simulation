import numpy as np
import matplotlib.pyplot as pl
import pickle
import random
import seaborn
import matplotlib as mpl
from PIL import Image, ImageFilter
from tifffile import tifffile
import seaborn as sns
import time
import math
import copy
import os
from os import path

from scipy.signal import find_peaks

import h5py
import pandas as pd
import sys
from typing import Optional, Dict, List, Tuple

from math import sqrt

mpl.rcParams['pdf.fonttype'] = 42


def set_journal_style(plot_width_mm=30, plot_height_mm=30, margin_mm=10):
    """
    Set Matplotlib style for journal publication with:
    - Only x and y axes visible.
    - A legend without a bounding box and with elements having only a fill (no stroke).
    - An actual plot area (excluding labels) of at least `plot_width_mm` × `plot_height_mm`.

    Parameters:
    - plot_width_mm (float): Minimum plot area width in mm (default: 30 mm).
    - plot_height_mm (float): Minimum plot area height in mm (default: 30 mm).
    - margin_mm (float): Extra margin for labels and titles (default: 10 mm).
    """
    # Convert mm to inches (1 inch = 25.4 mm)
    fig_width_in = (plot_width_mm + 2 * margin_mm) / 25.4
    fig_height_in = (plot_height_mm + 2 * margin_mm) / 25.4

    pl.rcParams.update({
        'font.size': 7,
        'font.family': 'Arial',
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'lines.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'savefig.dpi': 300,  # High resolution
        'figure.figsize': (fig_width_in, fig_height_in),
        'figure.constrained_layout.use': True  # Ensure proper layout
    })

    # Function to remove the top and right spines
    def remove_spines(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)

    # Function to format the legend (removes bounding box and legend stroke)
    def format_legend(ax):
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_linewidth(0)  # Remove bounding box
            legend.get_frame().set_facecolor("none")  # Transparent background
            legend.get_frame().set_edgecolor("none")  # No border

            # Modify legend elements to remove stroke but keep fill
            for handle in legend.legendHandles:
                try:
                    handle.set_edgecolor("none")  # Remove stroke
                except AttributeError:
                    pass  # Some elements may not support this, ignore errors

    return remove_spines, format_legend


def PositionToTemp(StartTemp, EndTemp, curr_pos, start_pos=0, end_pos=1464):
    arena_len = end_pos - start_pos
    return StartTemp + (curr_pos - start_pos) * (EndTemp - StartTemp) / arena_len


def ChamberTemps(exp_setup):
    if exp_setup.find("M") != -1:
        FrontTemp = int(exp_setup[exp_setup.find("F") + 1:exp_setup.find("_M")])
        MidTemp = int(exp_setup[exp_setup.find("M") + 1:exp_setup.find("_B")])
        BackTemp = int(exp_setup[exp_setup.find("B") + 1:])
    else:
        FrontTemp = int(exp_setup[exp_setup.find("F") + 1:exp_setup.find("_B")])
        BackTemp = int(exp_setup[exp_setup.find("B") + 1:])
        MidTemp = (FrontTemp + BackTemp) / 2
    if FrontTemp > 100:
        FrontTemp = FrontTemp / 10
    if BackTemp > 100:
        BackTemp = BackTemp / 10
    return FrontTemp, MidTemp, BackTemp


def MeanAngles(angle_vals, start_pos, end_pos):
    ang_means = []
    for i, j in zip(start_pos, end_pos):
        m = np.arange(np.max([i, 0]), np.min([j + 1, len(angle_vals)]))
        cosines = np.nanmean(np.cos(angle_vals[m]))
        sines = np.nanmean(np.sin(angle_vals[m]))
        ang_means.append(np.arctan2(sines,cosines))
    ang_means = np.array(ang_means)
    return (ang_means)

def find_bout_start_end_by_peak(instant_speed: np.ndarray, spd_thresh: float, pk_width: int,
                                delta_thresh: float) -> [np.ndarray, np.ndarray]:
    """
    Finds peaks in the speed trace and from there extends bouts forwards and backwards while the speed is decreasing
    :param instant_speed: The instand speed trace in which to identify bouts
    :param spd_thresh: The minimal peak height value in mm/s
    :param pk_width: The minimal peakd width in frames
    :param delta_thresh: From the peak bouts are extended while the speed is at least dropping by delta_thresh
    :return:
        [0]: n_bouts long vector of starts
        [1]: n_bouts long vector of ends
    """
    # First identify peaks
    peak_indices = find_peaks(instant_speed, height=spd_thresh, width=pk_width)[0]
    n_bouts = peak_indices.size
    starts = np.zeros(n_bouts, int) - 1
    ends = starts.copy()
    # From each peak walk forwards and backwards - stop walking if the average drop in speed across the next
    # five frames is smaller than the threshold
    for i, pi in enumerate(peak_indices):
        s = pi
        while True:
            delta = (instant_speed[s] - instant_speed[s-5]) / 5
            s -= 1
            if np.isnan(delta):
                s = -1
                break
            if delta < delta_thresh and instant_speed[s] < spd_thresh:
                break
        e = pi
        while True:
            if e + 5 >= instant_speed.size:
                break
            delta = (instant_speed[e] - instant_speed[e + 5]) / 5
            e += 1
            if np.isnan(delta):
                e = -1
                break
            if delta < delta_thresh and instant_speed[e] < spd_thresh:
                break
        starts[i] = s
        ends[i] = e
    # remove any pairs that have a -1 as this indicates that no valid value was found
    valid = np.logical_and(starts != -1, ends != -1)

    return starts[valid], ends[valid]


def validate_trajectories(trajectories, thresh=0.05):
    num_nan = np.sum(np.isnan(trajectories))
    l_traj = len(trajectories)
    if num_nan / l_traj > thresh:
        return np.array([])
    else:
        return trajectories


def remove_edges(experiments, front_dist=0, back_dist=0, left_dist=0, right_dist=0):
    if len(experiments) == 0:
        return experiments
    experiments[experiments[:, 0] < (0 + left_dist)] = np.nan
    experiments[experiments[:, 1] < (0 + front_dist)] = np.nan
    experiments[experiments[:, 0] > (330 - right_dist)] = np.nan
    experiments[experiments[:, 1] > (1464 - back_dist)] = np.nan

    return experiments


def VisualizeExperiment(track, exp="Unnamed", i="Unnumbered"):
    pl.figure()
    pl.scatter(track[:, 0], track[:, 1], s=0.5, alpha=0.1)
    pl.scatter(track[0, 0], track[0, 1], marker='*')
    pl.axis('equal')
    pl.title(f"Experiment {exp} {i}")


def fix_angle_trace(angles: np.ndarray, ad_thresh: float, max_ahead: int) -> np.ndarray:
    """
    Tries to fix sudden jumps in angle traces (angle differences that are implausible) by filling stretches between
    with the pre-jump angle
    :param angles: The angle trace to fix
    :param ad_thresh: The largest plausible delta-angle (arc distance on unit circle, i.e. smallest angular difference)
    :param max_ahead: The maximal amount of frames to look ahead for a jump back (longer stretches won't be fixed)
    :return: The corrected angle trace
    """
    adists = np.r_[0, arc_distance(angles)]
    angles_corr = np.full(angles.size, np.nan)
    index = 0
    while index < adists.size:
        ad = adists[index]
        if np.isnan(ad):
            angles_corr[index] = angles[index]
            index += 1
            continue
        if ad < ad_thresh:
            angles_corr[index] = angles[index]
            index += 1
        else:
            # start correction loop
            next_jump_ix = index + 1
            for i in range(max_ahead):
                if i + next_jump_ix >= adists.size:
                    # nothing we can do here, just set this one to NaN by not filling it and continue
                    break
                if adists[i + next_jump_ix] >= ad_thresh:
                    # we found a similar jump within the next ten frames fill with initial angle to correct
                    replace_angle = angles_corr[index - 1]
                    assert np.isfinite(replace_angle)
                    angles_corr[index:i + next_jump_ix + 1] = replace_angle
                    index = next_jump_ix + i
                    assert np.isfinite(angles_corr[index])
                    break
            index += 1
    return angles_corr


def edge_condition(curr_pos, xm, ym, angle):
    """
    Applies edge criterion for virtual fish swimming in a gradient of the same dimension as the chamber;
    If the fish exits the arena, it would be turned back to the chamber by reflection
    (Must change the parameter descriptions)
    :param angles: The angle trace to fix
    :param ad_thresh: The largest plausible delta-angle (arc distance on unit circle, i.e. smallest angular difference)
    :param max_ahead: The maximal amount of frames to look ahead for a jump back (longer stretches won't be fixed)
    :return: The corrected angle trace
    """
    boutlen = len(xm)
    if ym[-1] <= 0 or ym[-1] >= 1464:
        #         print(f"Applying edge condition on current position {curr_pos},{xm[-1]},{ym[-1]}")
        d = np.sqrt((xm[-1] - curr_pos[0]) ** 2 + (ym[-1] - curr_pos[1]) ** 2)
        #         print((732-curr_pos[1]))

        new_angle = np.arctan((732 - curr_pos[1]) / (160 - curr_pos[0]))
        # Accounting for smaller range of of inverse tangent function
        if curr_pos[0] > 160:
            new_angle = new_angle + np.pi
        x_dis = np.cos(new_angle) * d
        y_dis = np.sin(new_angle) * d
        x_move = curr_pos[0] + np.linspace(0, x_dis, boutlen + 1)[1:]
        y_move = curr_pos[1] + np.linspace(0, y_dis, boutlen + 1)[1:]
    #         print(f"new angle:{np.rad2deg(new_angle)},{x_move[-1]},{y_move[-1]}")

    else:
        x_move = xm
        y_move = ym
        new_angle = angle[0]
    return x_move, y_move, np.full(boutlen, new_angle)


def move_virtual_fish(curr_pos, curr_angle, turn_angle, distance, interbout, boutlen):

    new_angle=curr_angle + turn_angle
    if new_angle>np.pi:
        new_angle=new_angle-2*np.pi
    elif new_angle< -np.pi:
        new_angle=new_angle+2*np.pi
    bout_angle = np.full(boutlen+interbout, new_angle)


    x_move = np.cos(bout_angle[0]) * distance
    y_move = np.sin(bout_angle[0]) * distance

    x_movement = curr_pos[0] + np.linspace(0, x_move, boutlen + 1)[1:]
    y_movement = curr_pos[1] + np.linspace(0, y_move, boutlen + 1)[1:]

    x_movement = np.hstack([np.ones(interbout)*x_movement[0],x_movement])
    y_movement = np.hstack([np.ones(interbout)*y_movement[0],y_movement])
    #     x_movement,y_movement,bout_angle=edge_condition(curr_pos,x_movement,y_movement,bout_angle)

    bout_pos = np.vstack([x_movement, y_movement]).T
    bout_angle = bout_angle.T

    return bout_pos, bout_angle

def find_t_dt_bin(t, dt):
    # Assuming dt bins are -0.20 to 0.20 with binwidth of 0.1; 6 bins in total
    # Assuming t bins are 18 to 32 with binwidth of 2; 8 bins in total

    dt_bin = int((dt - (-0.20)) / 0.10) + 1
    t_bin = int((t - 18) / 2)

    if dt < -0.2:
        dt_bin = 0
    elif dt > 0.2:
        dt_bin = 5
    return t_bin, dt_bin

def FlipStates(turn_angles):
    last_state=turn_angles[0]
    current_state=turn_angles[0]
    flips=[]
    for i in turn_angles:
        current_state=i
        curr_flip=last_state*current_state
        flips.append(curr_flip)
        if curr_flip==-1:
            last_state=current_state
    flips=np.array(flips)
    return flips

def get_trajectories(positions,bout_start,bout_end):
    return np.hstack([positions[bout_start-10:bout_end+10]])

def normalizecolormap(start,end):
    return pl.Normalize(start,end)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)

#Sampling from a gaussian mixture model

def collect_trajectories(df: pd.DataFrame, frame_rate=100, n_seconds=2):
    """
    From one bout structure generates a list of bout structures with each structure representing
    one continuous trajectory

    :param df: A bout dataframe
    :param frame_rate: To convert bout distances into time for trajectory breaking
    :param n_seconds: Maximal delta-t between consecutive bout starts within a trajectory
    :return: List of trajectory bout dataframes
    """

    traj_breaks = np.r_[1, np.diff(df['Original index'])] > 1
    # add additional trajectory breaks for bouts that are more than n_seconds apart
    delta_time = np.r_[0, np.diff(df['Start'])] / frame_rate
    traj_breaks = np.logical_or(traj_breaks, delta_time > n_seconds)
    traj_starts = np.r_[0, np.where(traj_breaks)[0], traj_breaks.size]
    traj_list = []
    for i, ts in enumerate(traj_starts[:-1]):
        traj_list.append(df.iloc[ts:traj_starts[i + 1]])
    return traj_list


def FigurePlots():
    pl.xlabel("Temperatures ($^\circ$C)", font='Arial', size=15)
    pl.ylabel("Delta temperatures ($^\circ$C)", font='Arial', size=15)


def bootstrapped(df, qty, n_samples=200,List="None"):
    if List == "None":
        List = ['F16_B16', 'F18_B18', 'F20_B20', 'F22_B22', 'F24_B24', 'F26_B26', 'F28_B28', 'F30_B30', 'F32_B32',
                'F34_B34']
    mean_qty = np.zeros((10, n_samples))
    for n, exp in enumerate(List):
        t_df = df[exp]
        num_expt = len(t_df)
        for i in range(n_samples):
            sample = np.random.choice(list(t_df.keys()), size=num_expt, replace=True)
            temp_df = []
            for j in sample:
                temp_df.append(t_df[j][qty])
            combined_df = pd.concat(temp_df)
            if qty == "Absolute Angle change":
                combined_df = combined_df[combined_df > 5]
            mean_qty[n][i] = combined_df.mean()
    return mean_qty


def bootstrapped_gradient(df, group, qty, n_samples=200,List="None"):

    if List == "None":
        List = ['F26_B18', 'F32_B24']
    BootList = {}
    for exp in df:
        BootList[exp] = []
        for i in df[exp]:
            BootList[exp].append(i)
    combined={}

    for exp in df:
        combined[exp]={}
        for i in df[exp]:
            temp_df=df[exp][i]
            if qty == "Absolute Angle change":
                temp_df = temp_df[temp_df[qty] > 5]
            combined[exp][i]=temp_df.groupby(group)[qty].mean()

    boot_strapped = []
    for n in range(n_samples):
        sample = {"F26_B18": [], "F32_B24": []}
        for exp in List:
            sample[exp] = np.random.choice(BootList[exp], size=len(BootList[exp]), replace=True)
        sampled_df = []
        for exp in sample:
            for i in sample[exp]:
                sampled_df.append(combined[exp][i])
        sampled_df=np.vstack(sampled_df)

        boot_strapped.append(np.nanmean(sampled_df,axis=0))

    if group == "Temperature_bins":
        bootstrapped_df = pd.DataFrame(boot_strapped, columns=np.arange(19, 32, 2))
    elif group == "Delta T_bins":
        bootstrapped_df = pd.DataFrame(boot_strapped, columns=np.arange(-0.2, 0.25, 0.1))

    return bootstrapped_df

def bootstrapped_histogram(df, qty, bins, n_samples=200,density=True,List="None"):
    if List == "None":
        List = ['F16_B16', 'F18_B18', 'F20_B20', 'F22_B22', 'F24_B24', 'F26_B26', 'F28_B28', 'F30_B30', 'F32_B32',
                'F34_B34']
    mean_qty={}
    for exp in List:
        mean_qty[exp] = np.zeros((n_samples, len(bins)-1))
    for n, exp in enumerate(List):
        t_df = df[exp]
        num_expt = len(t_df)
        for i in range(n_samples):
            sample = np.random.choice(list(t_df.keys()), size=num_expt, replace=True)
            temp_df = []
            for j in sample:
                if qty=="Angle change":
                    temp_df.append(np.rad2deg(t_df[j][qty]))
                else:
                    temp_df.append(t_df[j][qty])
            combined_df = pd.concat(temp_df)
            mean_qty[exp][i] = np.histogram(combined_df.values,bins=bins,density=density)[0]
    return mean_qty

class Condition:
    name: str
    bout_df: dict
    ibi_list:dict
    angle_list:dict
    displacement_list:dict

    def __init__(self, name:str,bout_df: dict):
        self.name=name
        self.bout_df=bout_df

        self.ibi_list={}
        self.angle_list={}
        self.displacement_list={}

        if name == "gradient":
            T_range=np.arange(18,32.1,0.5)
            dT_range=np.arange(-0.25,0.26,0.05)

            for i,t in enumerate(T_range):
                    self.ibi_list[i]={}
                    self.angle_list[i]={}
                    self.displacement_list[i]={}

                    for j,dt in enumerate(dT_range):
                        self.ibi_list[i][j]=[]
                        self.angle_list[i][j]=[]
                        self.displacement_list[i][j]=[]

            for exp in bout_df:
                for i in bout_df[exp]:
                    for ind, bout in bout_df[exp][i].iterrows():
                        t_ind = int((bout["Temperature"] - 18) / 0.5)
                        dt_ind = int((bout["Prev Delta T"] - (-0.25)) / 0.05)
                        if dt_ind < 0:
                            dt_ind = 0
                        elif dt_ind > 10:
                            dt_ind = 10

                        if t_ind > 27:
                            t_ind = 27
                        elif t_ind < 0:
                            t_ind = 0

                        self.ibi_list[t_ind][dt_ind].append(bout["IBI"])
                        self.angle_list[t_ind][dt_ind].append(bout["Angle change"])
                        self.displacement_list[t_ind][dt_ind].append(bout["Displacement"])
        elif name=="constant":
            T_range = np.arange(18, 32.1, 2)

            for i, t in enumerate(T_range):
                self.ibi_list[i] = []
                self.angle_list[i] = []
                self.displacement_list[i] = []

            for exp in bout_df:
                for i in bout_df[exp]:
                    for ind,bout in bout_df[exp][i].iterrows():
                        t_ind=int((bout["Temperature"]-18)/2)

                        self.ibi_list[t_ind].append(bout["IBI"])
                        self.angle_list[t_ind].append(bout["Angle change"])
                        self.displacement_list[t_ind].append(bout["Displacement"])

    def pick_bout_by_t_dt(self,temp:float,d_temp:float)-> Tuple[float, float, float]:

        if self.name == "gradient":
            t_ind = int((temp - 18) / 0.5)
            dt_ind = int((d_temp - (-0.25)) / 0.05)
            if dt_ind < 0:
                dt_ind = 0
            elif dt_ind > 10:
                dt_ind = 10

            if t_ind > 27:
                t_ind = 27
            elif t_ind < 0:
                t_ind = 0
            ibi=random.choice(self.ibi_list[t_ind][dt_ind])
            angle=random.choice(self.angle_list[t_ind][dt_ind])
            disp = random.choice(self.displacement_list[t_ind][dt_ind])
        elif self.name=="constant":

            t_ind = int((temp - 18) / 2)
            t_refine=temp%2

            r=np.random.random()
            if r>t_refine:
                t_ind=t_ind+1

            ibi=random.choice(self.ibi_list[t_ind])
            angle=random.choice(self.angle_list[t_ind])
            disp = random.choice(self.displacement_list[t_ind])

        return ibi,angle,disp

class Experiment:
    name: str
    front: int
    back: int
    start_pos: int
    end_pos: int

    def __init__(self, name:str,front: int, back:int, start_pos: int,end_pos:int):
        self.name=name
        self.front=front
        self.back=back
        self.start_pos=start_pos
        self.end_pos=end_pos
