import argparse
import math

import gym
from collections import deque

import numpy as np

from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image2
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
import point
import pandas as pd



#caculate the speed
def calSpeed(position1,position2):
    x, y = position1
    x2, y2 = position2
    return math.sqrt((x-x2)**2+(y-y2)**2)

def isInTrack(position,trackList):

    x,y=position
    pp = point.Point(x, y)
    for i in range(len(trackList)):
        p1 = point.Point(trackList[i][0][0][0],trackList[i][0][0][1])
        p2 = point.Point(trackList[i][0][1][0],trackList[i][0][1][1])
        p3 = point.Point(trackList[i][0][2][0],trackList[i][0][2][1])
        p4 = point.Point(trackList[i][0][3][0],trackList[i][0][3][1])
        if point.IsPointInMatrix(p1, p2, p3, p4, pp):

            return True


    return False

def isInTrack2(position,trackList):

    x,y=position
    for i in range(len(trackList)):

        xc=(trackList[i][0][0][0]+trackList[i][0][1][0]+trackList[i][0][2][0]+trackList[i][0][3][0])/4
        yc=(trackList[i][0][0][1]+trackList[i][0][1][1]+trackList[i][0][2][1]+trackList[i][0][3][1])/4

        if (xc-5<x<xc+5)and (yc-5<y<yc+5):
            return True


    return False



def origin(path,epchos):



    #load models and init env

    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=False,default="models/trial_400.h5", help='The `.h5` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=epchos, help='The number of episodes should the model plays.')
    args = parser.parse_args()
    train_model = args.model
    play_episodes = args.episodes
    env = gym.make('CarRacing-v0')
    agent = CarRacingDQNAgent(epsilon=0) # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(train_model)



    # datas need to collect

    Speed=[]
    Ave_Color_Value=[]
    Frames=[]
    Safety=[]
    lastPosition = [(0,0)]
    e_number=[]
    end_number=[]
    end_label=[]

    guard = 0



    for e in range(play_episodes):
        print("the epoch is")
        print(e)

        if len(Frames)>100000:
            break


        frameCounter=0
        init_state = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1


        while True:
                if frameCounter==120:
                    print("ok")


                env.render()

                current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                action = agent.act(current_state_frame_stack)
                next_state, reward, done, info = env.step(action)
                posx,posy=info[0]
                lastPosition.append((posx,posy))




                if frameCounter!=0 :
                    Speed.append(calSpeed(lastPosition[frameCounter+1],lastPosition[frameCounter]))
                    print("speed is {}".format(Speed[frameCounter-1]))

                if frameCounter>10:
                    if Speed[frameCounter-1]==0:
                        for i in range(frameCounter):
                            end_number.append(frameCounter)
                            if i + 60 > frameCounter:
                                Safety.append(0)
                            else:
                                Safety.append(1)
                        end_label.append(1)
                        break
                print(info[0])


                #if out of track, then break
                if isInTrack(info[0],info[1])==False:
                    for i in range(frameCounter):
                        end_number.append(frameCounter)
                        if i+60>frameCounter:
                            Safety.append(0)
                        else:
                            Safety.append(1)
                    end_label.append(0)
                    break

                if frameCounter>300:
                    for i in range(frameCounter):
                        end_number.append(frameCounter)
                        Safety.append(1)
                    end_label.append(1)
                    break




                frameCounter=frameCounter+1
                print('total accuracy is {}, {} out of {},  total track is {}, reward is {}'.format(guard/frameCounter,guard,frameCounter,len(info[1]),total_reward))

                total_reward += reward
                print("frameCounter is {}".format(frameCounter))

                next_state, ave_pixel_value= process_state_image2(next_state,frameCounter,e)
                Frames.append(next_state)
                state_frame_stack_queue.append(next_state)
                e_number.append(e)
                Ave_Color_Value.append(ave_pixel_value)
                if done:
                    print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e+1, play_episodes, time_frame_counter, float(total_reward)))
                    for i in range(frameCounter):
                        end_number.append(frameCounter)
                        Safety.append(1)
                    break
                time_frame_counter += 1
                guard=guard+1

    print(len(Speed))
    print(len(Ave_Color_Value))
    print("------")
    print(len(Safety))
    print("------")
    print(e)
    # fr = np.array(Frames)
    safeties=np.array(Safety)
    # np.save("datas/safes.npy", fr)
    np.save("./videos/labels.npy", safeties)
    np.save("./videos/label2.npy", end_label)
    print(len(end_label))



    # dataframe = pd.DataFrame({'Speed': Speed,})
    dataframe = pd.DataFrame({ 'Ave_Color_Value': Ave_Color_Value,'Speed':Speed,'Safety': Safety})

    dataframe.to_csv("./videos/safes.csv",index=False,sep=',')
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="./datas")
    parser.add_argument('--epoch', type=int, default=100)
    args = parser.parse_args()

    origin(args.path,args.epoch)


