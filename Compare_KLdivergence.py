
from main_functions import *
from mm_bout_generator import *
import warnings
import re
from scipy.stats import entropy

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # colormap = pl.cm.RdBu_r  # You can choose any colormap you prefer
    # normalize = normalizecolormap(16, 34)
    # data_folder="/Users/kaarthikbalakrishnan/Library/CloudStorage/OneDrive-TheOhioStateUniversity/Desktop/Research/BehaviorExperiments/ZebraTrack/FishModelswithBouts"
    data_folder="/Users/kaarthikbalakrishnan/Library/CloudStorage/OneDrive-TheOhioStateUniversity/Desktop/Research/BehaviorExperiments/ZebraTrack/ColdBehaviorPaper/FishModelsBouts_250618"

    #Contains the raw experimental files
    file_list = sorted([f for f in os.listdir(data_folder) if ".fishexp" in f])

    #Defining the properties of different experiments in a dictionary for later use
    Gradients={"F26_B18":Experiment("F26_B18",26,18,0,1464),"F32_B24":Experiment("F32_B24",32,24,0,1464),"F32_B18":Experiment("F32_B18",32,18,0,2562),"F40_B10": Experiment("F40_B10", 40, 10, 0, 5490)}

    #Creating dictionaries to load positions, angles and ALL conditions are loaded into these variables
    Filt_vals={}
    tracked_angles={}
    for f in file_list:
        data_file = path.join(data_folder, f)
        myfile = open(data_file, "rb")
        print(f)
        Filt_vals[f[:f.find(".fishexp")]] = pickle.load(myfile)
        tracked_angles[f[:f.find(".fishexp")]] = pickle.load(myfile)
        myfile.close()
        print(f)

    # To plot experimental distribution of fish across experiments
    # Gradients=["F26_B18","F32_B24","F32_B18"]

    # Defining histograms and means to plot the histograms of temperatures; But first each experimental observation is converted into corresponding temperatures

    model_temp_hist = {}
    model_temp_mean={}
    Temperature_vals={}
    #First convert all values to positions to temperatures
    for exptype in Filt_vals:
        Temperature_vals[exptype]={}
        for gr, cond in zip(Gradients, Filt_vals[exptype]):

            Front = Gradients[gr].front
            Back = Gradients[gr].back
            start_pos = Gradients[gr].start_pos
            end_pos = Gradients[gr].end_pos
            exp = Gradients[gr].name

            print(cond, exp)
            Temperature_vals[exptype][exp] = {}

            if (len(Filt_vals[exptype][cond]) > 0):
                for j in Filt_vals[exptype][cond]:
                    Temperature_vals[exptype][exp][j] = np.vstack([Filt_vals[exptype][cond][j][:, 0],
                                                          PositionToTemp(Front, Back, Filt_vals[exptype][cond][j][:, 1],
                                                                         start_pos=start_pos, end_pos=end_pos)]).T

        #Plot histograms
        model_temp_hist[exptype]={}
        model_temp_mean[exptype]={}
        pl.figure()

        for gr in Gradients:
            temp_hist=[]

            Front = Gradients[gr].front
            Back = Gradients[gr].back
            start_pos = Gradients[gr].start_pos
            end_pos = Gradients[gr].end_pos

            if Front < Back:
                # bins = np.arange(Front + 0.1, Back, 0.1)
                x_range = np.arange(Back+0.5, Front-0.5, 0.14)
            else:
                # bins = np.arange(Back + 0.1, Front, 0.1)
                x_range = np.arange(Back+0.5, Front-0.5, 0.14)

            for i in Temperature_vals[gr]:
                temp_hist.append(np.histogram(Temperature_vals[gr][i][:, 1], bins=x_range)[0])
            print(exp)

            temp_hist = np.array(temp_hist)
            t_means = np.nanmean(temp_hist, axis=0)
            t_std = np.nanstd(temp_hist, axis=0) / np.sqrt(len(Temperature_vals[gr]))

            pl.errorbar((x_range[1:] + x_range[:-1]) / 2, t_means, yerr=t_std, label=f"{gr},n={len(Temperature_vals[gr])}",
                        alpha=0.3)
            pl.xlabel("Temperatures")
            pl.ylabel("Occupancy")
            pl.title(f"Fish model occupancy| {exptype}")

            model_temp_hist[exptype][gr]=temp_hist
            model_temp_mean[exptype][gr]=t_means
        pl.legend()

    #Now plotting for experimental distributions

    data_folder="/Users/kaarthikbalakrishnan/Desktop/Research/BehaviorExperiments/ZebraTrack/KAB_Pickles/Gradient"
    # bout_list = sorted([f for f in os.listdir(data_folder) if "bout.pkl" in f])
    # # fish_list= sorted([f for f in os.listdir(data_folder) if "fish.pkl" in f])
    #
    FileGroups=["F26_B18","F32_B24"]

    file_list = sorted([f for f in os.listdir(data_folder) if "fish.pkl" in f])
    bout_list = sorted([f for f in os.listdir(data_folder) if "bout.pkl" in f])

    FileDicts = {}
    BoutDicts = {}
    for fg in FileGroups:
        BoutDicts[fg] = [f for f in bout_list if fg in f]
        FileDicts[fg] = [f for f in file_list if fg in f]

    bout_grad_df = {}
    fish_data = {}

    for ftemp, btemp in zip(FileDicts, BoutDicts):
        ftemp_files = FileDicts[ftemp]
        btemp_files = BoutDicts[btemp]

        bout_grad_df[btemp] = {}
        fish_data[ftemp] = {}

        Front = Gradients[btemp].front
        Back = Gradients[btemp].back
        start_pos = Gradients[btemp].start_pos
        end_pos = Gradients[btemp].end_pos
        i = 0
        for tf, bf in zip(ftemp_files, btemp_files):
            data_file = path.join(data_folder, bf)
            myfile = open(data_file, "rb")
            avgs = pickle.load(myfile)
            bout_grad_df[btemp][i] = avgs
            myfile.close()

            data_file = path.join(data_folder, tf)
            myfile = open(data_file, "rb")
            avgs = pickle.load(myfile)
            fish_data[btemp][i] = avgs
            myfile.close()

            Temperature_vals[btemp][i] = avgs["Temperature"].values
            i = i + 1

    pl.figure()
    model_temp_hist["Experimental"]={}
    model_temp_mean["Experimental"]={}

    for exp in Temperature_vals:
        Front, Mid, Back = ChamberTemps(exp)
        x_range = np.arange(Back+0.5, Front-0.5, 0.14)
        temp_hist = []
        for i in Temperature_vals[exp]:
            # bdf = bout_grad_df[exp][i]
            temp_hist.append(np.histogram(Temperature_vals[exp][i], bins=x_range)[0])
        temp_hist = np.vstack(temp_hist)
        temp_mean = np.nanmean(temp_hist, axis=0)
        temp_std = np.nanstd(temp_hist, axis=0) / np.sqrt(len(bout_grad_df[exp]))
        pl.errorbar((x_range[0:-1] + x_range[1:]) / 2, temp_mean, yerr=temp_std,
                    label=f"{exp}; n={len(bout_grad_df[exp])}")
        model_temp_hist["Experimental"][exp]=temp_hist
        model_temp_mean["Experimental"][exp]=temp_mean
    pl.legend()
    pl.xlabel("Temperature")
    pl.ylabel("Density +/- se")
    pl.title("Experimental distribution")

    # pl.figure()
    # for exptype in model_temp_hist:
    #     for exp in model_temp_hist[exptype]:
    #         Front, Mid, Back = ChamberTemps(exp)
    #         x_range = np.arange(Back + 0.1, Front, 0.1)
    #
    #         temp_mean = np.nanmean(temp_hist, axis=0)
    #         temp_std = np.nanstd(temp_hist, axis=0) / np.sqrt(len(model_temp_hist[exptype][exp]))
    #         pl.errorbar((x_range[0:-1] + x_range[1:]) / 2, temp_mean, yerr=temp_std,
    #                     label=f"{exp}; n={len(model_temp_hist[exptype][exp])}")
    # pl.legend()

# KL_divergence=[]
    ExpTemps=["F26_B18","F32_B24"]
    for exptype in model_temp_mean:
        pl.figure()
        for exp in ExpTemps:
            Front, Mid, Back = ChamberTemps(exp)
            x_range = np.arange(Back+0.5, Front-0.5, 0.14)
            P=model_temp_mean["Experimental"][exp]
            Q=model_temp_mean[exptype][exp]
            kl_divergence = entropy(P, Q, base=np.e)  # Use natural log as the base
            # print(f"KL Divergence (D_KL(P || Q)): {kl_divergence}")
            pl.plot((x_range[0:-1] + x_range[1:]) / 2, model_temp_mean[exptype][exp],label=f"{exp}; n={len(model_temp_hist[exptype][exp])}|KL_divergence:{np.round(kl_divergence,3)}")
            # pl.plot((x_range[0:-1] + x_range[1:]) / 2, P,label=f"Experimental |KL_divergence:{kl_divergence}")

        pl.title(f"Fish model occupancy| {exptype}")
        pl.legend()




