# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from mainfunctions import *
from mm_bout_generator import *

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


if __name__ == '__main__':


    # for order in range(4):
    for order in [False,True]:
        if order:
            used_model='0_order'
        else:
            used_model='transition'
        Filt_vals = {}
        tracked_angles = {}

        bout_dataframe={}

        Gradients={"F26_B18":Experiment("F26_B18",26,18,0,1464),"F32_B24":Experiment("F32_B24",32,24,0,1464)}

        for gr in Gradients:
            Filt_vals[f"fishmodel_{Gradients[gr].name}"]  = {}
            tracked_angles[f"fishmodel_{Gradients[gr].name}"] = {}
            bout_dataframe[f"fishmodel_{Gradients[gr].name}"] = {}

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

                # obj = SwimBoutGenerator(
                #     "/Users/kaarthikbalakrishnan/Desktop/Research/BehaviorExperiments/ZebraTrack/sim_model_store_multi_order.hdf5", 0,0,2, order)

                #For the simpler model with just 4 parameters (t,dt,abs(dt) and t*dt)
                obj = SwimBoutGenerator("/Users/kaarthikbalakrishnan/Desktop/Research/BehaviorExperiments/ZebraTrack/sim_model_store.hdf5", 0,0,2, order)

                df_elements=[]
                while it < 180000:

                    #Pick new bout
                    next_bout = obj.draw_next_bout(temp, d_temp,force_mode_0=False)
                    ibi=int(next_bout.pre_bout_ibi_ms/10) ##Convert to 100 Hz for ease of simulation
                    dist=next_bout.bout_displacement_mm * 7
                    angle=np.deg2rad(next_bout.bout_angle_deg)
                    count=count+1

                    #Move fish to new position
                    bout_pos, bout_angle = move_virtual_fish(curr_pos, curr_angle, angle, dist, ibi, boutlen)

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

                    df_elements.append([next_bout.swim_mode,count,it,it+boutlen+ibi,dist,angle,ibi,temp,d_temp,delta_xy[0],delta_xy[1],curr_pos[0],curr_pos[1],curr_angle])

                    curr_pos = bout_pos[-1]
                    curr_angle = bout_angle[0]

                    d_temp = PositionToTemp(Front, Back, curr_pos[1], start_pos=start_pos, end_pos=end_pos) - temp

                    temp = PositionToTemp(Front, Back, curr_pos[1], start_pos=start_pos, end_pos=end_pos)

                    it = it + boutlen + ibi


                bout_dataframe[f"fishmodel_{Gradients[gr].name}"][num] = pd.DataFrame(df_elements,
                    columns=["State", "Original Index", "Start", "Stop", "Displacement", "Angle change", "IBI",
                             "Temperature", "Prev Delta T", "Delta X", "Delta Y", "X Position", "Y Position",
                             "Heading"])

                print(gr,num)
                Filt_vals[f"fishmodel_{Gradients[gr].name}"][num] = np.vstack(grad_pos)[12000:1000000]
                tracked_angles[f"fishmodel_{Gradients[gr].name}"][num] = np.hstack(grad_angle)[12000:1000000]

                del obj

        pickle_file = open(f"Fish_model_{used_model}_250618.fishexp", "wb")
        pickle.dump(Filt_vals, pickle_file)
        pickle.dump(tracked_angles, pickle_file)
        pickle_file.close()

        pickle_file = open(f"Fish_model_{used_model}_bouts_250618.pkl", "wb")
        pickle.dump(bout_dataframe, pickle_file)
        pickle_file.close()

#"Full" experiment simulation; F32_B18


    for order in [False,True]:
        if order:
            used_model='0_order'
        else:
            used_model='transition'

        Filt_vals = {}
        tracked_angles = {}

        bout_dataframe = {}

        Gradients = {"F32_B18":Experiment("F32_B18", 32, 18, 0, 2562)}

        # Gradients = {"F40_B10": Experiment("F40_B10", 40, 10, 0, 5490)}

        for gr in Gradients:
            Filt_vals[f"fishmodel_{Gradients[gr].name}"] = {}
            tracked_angles[f"fishmodel_{Gradients[gr].name}"] = {}
            bout_dataframe[f"fishmodel_{Gradients[gr].name}"] = {}

            for num in range(200):

                start_pos = Gradients[gr].start_pos
                end_pos = Gradients[gr].end_pos

                first_pos = [np.random.randint(0, 300), np.random.randint(0, end_pos)]
                Front = Gradients[gr].front
                Back = Gradients[gr].back

                curr_pos = first_pos
                curr_angle = (np.random.rand() - 0.5) * 2 * np.pi

                temp = PositionToTemp(Front, Back, curr_pos[1], start_pos=start_pos, end_pos=end_pos)
                d_temp = 0

                grad_pos = []
                grad_angle = []
                it = 0
                # bout_pos = [[-25, -40]]
                th = []
                dth = []
                boutlen = 20
                count = 0

                # obj = SwimBoutGenerator(
                #     "/Users/kaarthikbalakrishnan/Desktop/Research/BehaviorExperiments/ZebraTrack/sim_model_store_multi_order.hdf5",
                #     0, 0, 2, order)

                obj = SwimBoutGenerator(
                    "/Users/kaarthikbalakrishnan/Desktop/Research/BehaviorExperiments/ZebraTrack/sim_model_store.hdf5",
                    0, 0, 2, order)

                df_elements = []
                while it < 180000:

                    # Pick new bout
                    next_bout = obj.draw_next_bout(temp, d_temp, force_mode_0=False)
                    ibi = int(next_bout.pre_bout_ibi_ms / 10)  ##Convert to 100 Hz for ease of simulation
                    dist = next_bout.bout_displacement_mm * 7
                    angle = np.deg2rad(next_bout.bout_angle_deg)
                    count = count + 1

                    # Move fish to new position
                    bout_pos, bout_angle = move_virtual_fish(curr_pos, curr_angle, angle, dist, ibi, boutlen)

                    ##Check y-direction
                    if bout_pos[-1][1] < 0:
                        bout_pos[:, 1] = np.where(bout_pos[:, 1] < 0, 0, bout_pos[:, 1])
                        bout_angle = np.where(bout_pos[:, 1] < 0, np.arctan2(0, bout_pos[-1][0] - curr_pos[0]), bout_angle)
                    elif bout_pos[-1][1] > end_pos:
                        bout_pos[:, 1] = np.where(bout_pos[:, 1] > end_pos, end_pos, bout_pos[:, 1])
                        bout_angle = np.where(bout_pos[:, 1] > end_pos, np.arctan2(0, bout_pos[-1][0] - curr_pos[0]),
                                              bout_angle)

                    ###Check x-direction
                    if bout_pos[-1][0] < 0:
                        bout_pos[:, 0] = np.where(bout_pos[:, 0] < 0, 0, bout_pos[:, 0])
                        bout_angle = np.where(bout_pos[:, 0] < 0, np.arctan2(bout_pos[-1][1] - curr_pos[1], 0), bout_angle)
                    elif bout_pos[-1][0] > 320:
                        bout_pos[:, 0] = np.where(bout_pos[:, 0] > 320, 320, bout_pos[:, 0])
                        bout_angle = np.where(bout_pos[:, 0] > 320, np.arctan2(bout_pos[-1][1] - curr_pos[1], 0),
                                              bout_angle)

                    grad_pos.append(bout_pos)
                    grad_angle.append(bout_angle)

                    delta_xy = bout_pos[-1] - curr_pos

                    df_elements.append(
                        [next_bout.swim_mode, count, it, it + boutlen + ibi, dist, angle, ibi, temp, d_temp, delta_xy[0],
                         delta_xy[1], curr_pos[0], curr_pos[1], curr_angle])

                    curr_pos = bout_pos[-1]
                    curr_angle = bout_angle[0]

                    d_temp = PositionToTemp(Front, Back, curr_pos[1], start_pos=start_pos, end_pos=end_pos) - temp

                    temp = PositionToTemp(Front, Back, curr_pos[1], start_pos=start_pos, end_pos=end_pos)

                    it = it + boutlen + ibi

                bout_dataframe[f"fishmodel_{Gradients[gr].name}"][num] = pd.DataFrame(df_elements,
                                                                                      columns=["State", "Original Index",
                                                                                               "Start", "Stop",
                                                                                               "Displacement",
                                                                                               "Angle change", "IBI",
                                                                                               "Temperature",
                                                                                               "Prev Delta T", "Delta X",
                                                                                               "Delta Y", "X Position",
                                                                                               "Y Position",
                                                                                               "Heading"])

                print(gr, num)
                Filt_vals[f"fishmodel_{Gradients[gr].name}"][num] = np.vstack(grad_pos)[12000:1000000]
                tracked_angles[f"fishmodel_{Gradients[gr].name}"][num] = np.hstack(grad_angle)[12000:1000000]

                del obj
        pickle_file = open(f"Fish_model_fullsim_{used_model}_250618.fishexp", "wb")
        pickle.dump(Filt_vals, pickle_file)
        pickle.dump(tracked_angles, pickle_file)
        pickle_file.close()

        pickle_file = open(f"Fish_model_fullsim_{used_model}_bouts_250618.pkl", "wb")
        pickle.dump(bout_dataframe, pickle_file)
        pickle_file.close()
