# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from main_functions import *
# from mm_bout_generator_2ndorder import *
from mm_bout_generator import *
import warnings
import re

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

    # For constant experiment simulations
    # data_folder = "/Users/kaarthikbalakrishnan/Desktop/Research/BehaviorExperiments/ZebraTrack/KAB_Pickles/Constant/"
    # cond='constant'
    # expts='F32_B32,F30_B30,F28_B28,F26_B26,F24_B24,F22_B22,F20_B20,F18_B18' # For constant experiment (T) simulations

    #For gradient experiment (TdT) simulations, uncomment this
    data_folder = "/Users/kaarthikbalakrishnan/Desktop/Research/BehaviorExperiments/ZebraTrack/KAB_Pickles/Gradient/"
    cond='gradient'
    expts='F26_B18,F32_B24' # For gradient experiment (TdT) simulations

    FileGroups = list(expts.split(','))
    file_list = sorted([f for f in os.listdir(data_folder) if "fish.pkl" in f])
    bout_list = sorted([f for f in os.listdir(data_folder) if "bout.pkl" in f])
    if len(bout_list) ==len(file_list):
        print("Bout files present, loading...")

        FileDicts = {}
        BoutDicts={}
        for fg in FileGroups:
            BoutDicts[fg] = [f for f in bout_list if fg in f]
            FileDicts[fg] = [f for f in file_list if fg in f]

        bout_df = {}
        fish_data = {}
        for ftemp,btemp in zip(FileDicts,BoutDicts):
            ftemp_files = FileDicts[ftemp]
            btemp_files = BoutDicts[btemp]

            bout_df[btemp] = {}
            fish_data[ftemp] = {}
            i = 0
            for tf,bf in zip(ftemp_files,btemp_files):
                print(ftemp,i)
                data_file = path.join(data_folder, bf)
                with open(data_file, "rb") as myfile:
                    avgs = pickle.load(myfile)
                    bout_df[btemp][i] = avgs

                data_file = path.join(data_folder, tf)
                with open(data_file, "rb") as myfile:
                    avgs = pickle.load(myfile)
                    fish_data[btemp][i] = avgs
                i=i+1

    VirtualFish = Condition(cond, bout_df)

    Filt_vals = {}
    tracked_angles = {}
    bout_dataframe = {}

    print("Ready to start simulations")
    Gradients={"F26_B18":Experiment("F26_B18",26,18,0,1464),"F32_B24":Experiment("F32_B24",32,24,0,1464)}

    for gr in Gradients:
        Filt_vals[f"experimentfishmodel_{Gradients[gr].name}"]  = {}
        tracked_angles[f"experimentfishmodel_{Gradients[gr].name}"] = {}
        bout_dataframe[f"experimentfishmodel_{Gradients[gr].name}"] = {}

        for num in range(200):

            start_pos=Gradients[gr].start_pos
            end_pos=Gradients[gr].end_pos

            first_pos = [np.random.randint(0, 300), np.random.randint(0, end_pos)]
            Front=Gradients[gr].front
            Back=Gradients[gr].back

            curr_pos = first_pos
            curr_angle = (np.random.rand() - 0.5) * 2 * np.pi

            temp = PositionToTemp(Front, Back, curr_pos[1], start_pos=start_pos, end_pos=end_pos)
            d_temp=0

            grad_pos = []
            grad_angle = []
            it = 0
            # bout_pos = [[-25, -40]]
            th = []
            dth = []
            boutlen = 20
            count=0
            df_elements = []
            state=0

            while it < 180000:

                ibi, angle, disp = VirtualFish.pick_bout_by_t_dt(temp,d_temp)
                ibi=int(ibi/10)
                disp=disp*7

                count=count+1

                bout_pos, bout_angle = move_virtual_fish(curr_pos, curr_angle, angle, disp, ibi, boutlen)
                # print(len(bout_pos),len(bout_angle))

                ##Check y-direction
                if bout_pos[-1][1]<0:
                    bout_pos[:,1]=np.where(bout_pos[:,1]<0,0,bout_pos[:,1])
                    bout_angle=np.where(bout_pos[:,1]<0,np.arctan2(0,bout_pos[-1][0]-curr_pos[0]),bout_angle)
                elif bout_pos[-1][1]>end_pos:
                    bout_pos[:,1]=np.where(bout_pos[:,1]>end_pos,end_pos,bout_pos[:,1])
                    bout_angle=np.where(bout_pos[:,1]>end_pos,np.arctan2(0,bout_pos[-1][0]-curr_pos[0]),bout_angle)

                ###Check x-direction
                if bout_pos[-1][0]<0:
                    bout_pos[:,0]=np.where(bout_pos[:,0]<0,0,bout_pos[:,0])
                    bout_angle=np.where(bout_pos[:,0]<0,np.arctan2(bout_pos[-1][1]-curr_pos[1],0),bout_angle)
                elif bout_pos[-1][0]>320:
                    bout_pos[:,0]=np.where(bout_pos[:,0]>320,320,bout_pos[:,0])
                    bout_angle=np.where(bout_pos[:,0]>320,np.arctan2(bout_pos[-1][1]-curr_pos[1],0),bout_angle)

                grad_pos.append(bout_pos)
                grad_angle.append(bout_angle)

                delta_xy = bout_pos[-1] - curr_pos

                # bout_idx=len(bout_dataframe[f"fishmodel_{Gradients[gr].name}"][num])
                df_elements.append(
                    [state, count, it, it + boutlen + ibi, disp, angle, ibi, temp, d_temp, delta_xy[0],
                     delta_xy[1], curr_pos[0], curr_pos[1], curr_angle])

                curr_pos = bout_pos[-1]
                curr_angle = bout_angle[0]
                it = it + boutlen + ibi
                # print(it,boutlen,ibi)

                d_temp = PositionToTemp(Front, Back, curr_pos[1], start_pos=start_pos, end_pos=end_pos) - temp

                temp = PositionToTemp(Front, Back, curr_pos[1], start_pos=start_pos, end_pos=end_pos)


                # if it % 10000 == 0:
                #     print(it)
            print(gr,num)
            bout_dataframe[f"experimentfishmodel_{Gradients[gr].name}"][num] = pd.DataFrame(df_elements,
                                                                                  columns=["State", "Original Index",
                                                                                           "Start", "Stop",
                                                                                           "Displacement",
                                                                                           "Angle change", "IBI",
                                                                                           "Temperature",
                                                                                           "Prev Delta T", "Delta X",
                                                                                           "Delta Y", "X Position",
                                                                                           "Y Position",
                                                                                           "Heading"])
            Filt_vals[f"experimentfishmodel_{Gradients[gr].name}"][num] = np.vstack(grad_pos)[12000:180000]
            tracked_angles[f"experimentfishmodel_{Gradients[gr].name}"][num] = np.hstack(grad_angle)[12000:180000]


    # pickle_file = open(f"Experiment_Fish_model_{cond}.fishexp", "wb")
    # pickle.dump(Filt_vals, pickle_file)
    # pickle.dump(tracked_angles, pickle_file)
    # pickle_file.close()
    #
    # pickle_file = open(f"Experiment_Fish_model_{cond}_bouts.pkl", "wb")
    # pickle.dump(bout_dataframe, pickle_file)
    # pickle_file.close()


