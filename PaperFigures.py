
from mainfunctions import *
import warnings

# Press the green button in the gutter to run the sPcript.
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Load constant temperature experiments
    data_folder = "/Users/kaarthikbalakrishnan/Desktop/Research/BehaviorExperiments/ZebraTrack/KAB_Pickles/Constant/"
    ConstantGroups = ["F34_B34","F32_B32","F30_B30","F28_B28","F26_B26","F24_B24","F22_B22","F20_B20","F18_B18","F16_B16"]
    bout_list = sorted([f for f in os.listdir(data_folder) if "bout.pkl" in f])
    file_list = sorted([f for f in os.listdir(data_folder) if "fish.pkl" in f])

    BoutDicts={}
    FileDicts = {}
    for fg in ConstantGroups:
        BoutDicts[fg] = [f for f in bout_list if fg in f]
        FileDicts[fg] = [f for f in file_list if fg in f]

    bout_df = {}
    fish_data = {}
    for ftemp, btemp in zip(FileDicts, BoutDicts):
        ftemp_files = FileDicts[ftemp]
        btemp_files = BoutDicts[btemp]

        bout_df[btemp] = {}
        fish_data[btemp]={}
        i = 0
        for tf, bf in zip(ftemp_files, btemp_files):
            data_file = path.join(data_folder, bf)
            with open(data_file,"rb") as myfile:
                avgs = pickle.load(myfile)
                bout_df[btemp][i] = avgs

            data_file = path.join(data_folder, tf)
            with open(data_file,'rb') as myfile:

                avgs = pickle.load(myfile)
                fish_data[btemp][i] = avgs
            i = i + 1
        print(btemp)

    #Gradient Experiments

    data_folder = "/Users/kaarthikbalakrishnan/Desktop/Research/BehaviorExperiments/ZebraTrack/KAB_Pickles/Gradient/"
    GradientGroups = ["F26_B18","F32_B24"]
    bout_list = sorted([f for f in os.listdir(data_folder) if "bout.pkl" in f])
    file_list = sorted([f for f in os.listdir(data_folder) if "fish.pkl" in f])

    BoutDicts = {}
    FileDicts = {}
    for fg in GradientGroups:
        BoutDicts[fg] = [f for f in bout_list if fg in f]
        FileDicts[fg] = [f for f in file_list if fg in f]

    bout_grad_df = {}
    for ftemp, btemp in zip(FileDicts, BoutDicts):
        ftemp_files = FileDicts[ftemp]
        btemp_files = BoutDicts[btemp]

        bout_df[btemp] = {}
        fish_data[btemp]={}
        i = 0
        for tf, bf in zip(ftemp_files, btemp_files):
            data_file = path.join(data_folder, bf)
            with open(data_file, "rb") as myfile:
                avgs = pickle.load(myfile)
                bout_df[btemp][i] = avgs

            data_file = path.join(data_folder, tf)
            with open(data_file, 'rb') as myfile:
                avgs = pickle.load(myfile)
                fish_data[btemp][i] = avgs
            i = i + 1
        print(btemp)

    # t_range = np.arange(19, 31.1, 2)
    # tbin_edges = [t - 1 for t in t_range] + [t_range[-1] + 1]  # Create edges
    #
    # dt_range = np.arange(-0.2, 0.25, 0.1)
    # dtbin_edges = [dt - 0.05 for dt in dt_range] + [dt_range[-1] + 0.05]  # Create edges
    #
    # for exp in bout_grad_df:
    #     print(exp)
    #     for i in bout_grad_df[exp]:
    #         bout_grad_df[exp][i]['Absolute Angle change'] = np.rad2deg(np.abs(bout_grad_df[exp][i]['Angle change']))
    #         bout_grad_df[exp][i]['Temperature_bins'] = pd.cut(bout_grad_df[exp][i]['Temperature'], bins=tbin_edges,
    #                                                           labels=t_range)
    #         bout_grad_df[exp][i]['Delta T_bins'] = pd.cut(bout_grad_df[exp][i]['Prev Delta T'], bins=dtbin_edges,
    #                                                       labels=dt_range)
    #
    # x_range = np.arange(18, 26.1, 1)
    # bin_edges = [x - 0.5 for x in x_range] + [x_range[-1] + 0.5]  # Create edges
    #
    # # Bin the data using pd.cut
    #
    # for exp in bout_df:
    #     print(exp)
    #     F = ChamberTemps(exp)[0]
    #     boot_strapped = []
    #     for i in bout_df[exp]:
    #         bout_df[exp][i]['Absolute Angle change'] = np.rad2deg(np.abs(bout_df[exp][i]['Angle change']))
    #         bout_df[exp][i]['Y Temperatures'] = PositionToTemp(18, 26, bout_df[exp][i]['Y Position'], 0, 203)
    #         bout_df[exp][i]['Y Temperature_bins'] = pd.cut(bout_df[exp][i]['Y Temperatures'], bins=bin_edges,
    #                                                        labels=x_range)


    print("Start plotting figures")
    ##### START PLOTTING FIGURES ##########

    PlotList=["F16_B16","F26_B26","F34_B34"]
    ColorList={"F16_B16":'blue',"F26_B26":'black',"F34_B34":'red'}
    remove_spines,format_legend=set_journal_style()

    #Figure S1A; IBI distribution
    x_range = np.arange(0, 1500, 10)
    qty="IBI"
    fig, ax = pl.subplots()

    bhist=bootstrapped_histogram(bout_df,qty,x_range,n_samples=200,density=True)
    for exp in PlotList:
        Front=ChamberTemps(exp)[0]
        boot_means = np.mean(bhist[exp], axis=0)
        ax.plot(x_range[:-1], boot_means,color=ColorList[exp])
        std_err = np.std(bhist[exp], axis=0)
        ax.fill_between(x_range[:-1], boot_means + std_err, boot_means - std_err,color=ColorList[exp], alpha=0.2,
                        label=f"{Front}C")
    ax.legend(fontsize=7)
    format_legend(ax)
    remove_spines(ax)
    ax.set_xlabel(f"{qty}")
    ax.set_ylabel("Density +/- se")
    # pl.savefig(f"S1_{qty}_dist.pdf",transparent=True,format="pdf", bbox_inches="tight")

    #Figure 1C; Displacement distribution
    x_range = np.arange(0, 10, 0.1)
    qty="Displacement"
    fig, ax = pl.subplots()

    bhist=bootstrapped_histogram(bout_df,qty,x_range,n_samples=200,density=True)
    for exp in PlotList:
        Front=ChamberTemps(exp)[0]
        boot_means = np.mean(bhist[exp], axis=0)
        ax.plot(x_range[:-1], boot_means,color=ColorList[exp])
        std_err = np.std(bhist[exp], axis=0)
        ax.fill_between(x_range[:-1], boot_means + std_err, boot_means - std_err,color=ColorList[exp], alpha=0.2,
                        label=f"{Front}C")
    ax.legend(fontsize=7)
    format_legend(ax)
    remove_spines(ax)
    ax.set_xlabel(f"{qty}")
    ax.set_ylabel("Density +/- se")
    # pl.savefig(f"F1_{qty}_dist.pdf",transparent=True,format="pdf", bbox_inches="tight")

    # Figure 1D; Turn angle distribution
    x_range = np.arange(-150, 150, 1)
    qty="Angle change"
    fig, ax = pl.subplots()
    bhist=bootstrapped_histogram(bout_df,qty,x_range,n_samples=200,density=True)
    for exp in PlotList:
        Front=ChamberTemps(exp)[0]
        boot_means = np.mean(bhist[exp], axis=0)
        pl.plot(x_range[:-1], boot_means,color=ColorList[exp])
        std_err = np.std(bhist[exp], axis=0)
        pl.fill_between(x_range[:-1], boot_means + std_err, boot_means - std_err,color=ColorList[exp], alpha=0.2,
                        label=f"{Front}C")
    ax.legend(fontsize=7)
    format_legend(ax)
    remove_spines(ax)
    ax.set_xlabel(f"Turn angle")
    ax.set_ylabel("Density +/- se")
    # pl.savefig(f"F1_{qty}_dist.pdf",transparent=True,format="pdf", bbox_inches="tight")

    #Figure 1B example trajectory and Figure 3C Reversal Maneuver
    exp = "F26_B18"
    i = 2
    track = fish_data[exp][i][["X Position", "Y Position"]].values
    track = track[::10]
    cold_turns = track[track[:, 1] > 130]
    cold_turns = cold_turns[np.logical_and(cold_turns[:, 0] < 20, cold_turns[:, 0] > 0)]
    j = 1200
    fig, ax = pl.subplots()
    pl.scatter(cold_turns[j:j + 200, 0], cold_turns[j:j + 200, 1], color='k', s=1, alpha=1, rasterized=True)
    pl.axis('equal')
    pl.title(j)
    # pl.savefig("F3_ColdAvoidance_looptrajectory.svg",dpi=600,bbox_inches='tight')

    fig, ax = pl.subplots()
    pl.scatter(track[:, 0], track[:, 1], color='b', s=0.2, alpha=0.1, rasterized=True)
    pl.axis('equal')
    pl.title(j)
    # pl.savefig("F1_Experiment_example.svg",dpi=600,bbox_inches='tight')



    #Figure 5E Cold experiment example
    exp = "F26_B18"
    i = 13

    track = fish_data[exp][i][["X Position", "Y Position"]].values
    track = track[::10]     ## Down sampling for easy plotting
    fig, ax = pl.subplots()
    pl.scatter(track[:, 0], track[:, 1], color='b', s=0.2, alpha=0.1, rasterized=True)
    pl.axis('equal')
    pl.title(i)
    # pl.savefig("F5E_ColdAvoidance_Experiment_example.svg",dpi=600,bbox_inches='tight')

    # Figure 5E Hot experiment example
    exp = "F32_B24"
    i = 2

    track = fish_data[exp][i][["X Position", "Y Position"]].values
    track = track[::10]  ## Down sampling for easy plotting
    fig, ax = pl.subplots()
    pl.scatter(track[:, 0], track[:, 1], color='r', s=0.2, alpha=0.1, rasterized=True)
    pl.axis('equal')
    pl.title(i)
    # pl.savefig("F5E_HotAvoidance_Experiment_example.svg",dpi=600,bbox_inches='tight')


    data_folder="/Users/kaarthikbalakrishnan/Library/CloudStorage/OneDrive-TheOhioStateUniversity/Desktop/Research/BehaviorExperiments/ZebraTrack/ColdBehaviorPaper/FishModelsBouts_Latest"
    file='Fish_model_3.fishexp'

    # Gradients={"F26_B18":Experiment("F26_B18",26,18,0,1464),"F32_B24":Experiment("F32_B24",32,24,0,1464)}

    simulation_data={}
    tracked_angles={}

    data_file = path.join(data_folder, file)
    with open(data_file, "rb") as myfile:
        print(file)
        simulation_data= pickle.load(myfile)


    # Figure 5E Cold experiment Model 3 simulation example

    exp = "fishmodel_F26_B18"
    i = 3

    track = simulation_data[exp][i]
    track = track[::10]  ## Down sampling for easy plotting
    fig, ax = pl.subplots()
    pl.scatter(track[:, 0], track[:, 1], color='b', s=0.2, alpha=0.1, rasterized=True)
    pl.axis('equal')
    pl.title(i)
    # # pl.savefig("F5E_ColdAvoidance_Model3_example.svg",dpi=600,bbox_inches='tight')


    # # Figure 5E Hot experiment Model 3 simulation example
    exp = "fishmodel_F32_B24"
    i = 13

    track = simulation_data[exp][i]
    track = track[::10]  ## Down sampling for easy plotting
    fig, ax = pl.subplots()
    pl.scatter(track[:, 0], track[:, 1], color='r', s=0.2, alpha=0.1, rasterized=True)
    pl.axis('equal')
    pl.title(i)
    # # pl.savefig("F5E_ColdAvoidance_Model3_example.svg",dpi=600,bbox_inches='tight')

    # Figure S5A Temperature stimulus presented to fish

    fig, ax = pl.subplots()
    monitor = np.loadtxt(
        '/Users/kaarthikbalakrishnan/Library/CloudStorage/OneDrive-TheOhioStateUniversity/Desktop/Research/BehaviorExperiments/ZebraTrack/241106_Fish95_GCaMP_HB_6dpf_FlowExperiment_warner_temp_stim_Z_0.temp')
    controller = np.loadtxt(
        '/Users/kaarthikbalakrishnan/Library/CloudStorage/OneDrive-TheOhioStateUniversity/Desktop/Research/BehaviorExperiments/ZebraTrack/241106_Fish95_GCaMP_HB_6dpf_FlowExperiment_warner_temp_stim_Z_0.c_temp')
    monitor_avg = [np.mean(monitor[i:i + 20]) for i in range(0, len(monitor), 20)]
    controller_avg = [np.mean(controller[i:i + 20]) for i in range(0, len(controller), 20)]
    ax.plot(monitor_avg, linestyle='-', color='orange', label="Monitor")
    ax.plot(controller_avg, linestyle='--', color='k', alpha=0.5, label="Controller")
    ax.axvline(720, linestyle='--', color='k')
    pl.xticks(np.arange(0, 1081, 360))
    pl.xlabel("Time [s]")
    pl.ylabel("Temperature [C]")
    pl.legend()
    format_legend(ax)
    remove_spines(ax)
    #pl.savefig("Temperature_stim.pdf", transparent=True, format="pdf", bbox_inches="tight")

