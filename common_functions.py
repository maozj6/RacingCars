import cv2
import numpy as np
from PIL import Image


def drawTrack(trackList):

    img=np.zeros((600,600))
    for i in range(len(trackList)):

        p1 = [int(trackList[i][0][0][0])+300, int(trackList[i][0][0][1])+300]
        p2 = [int(trackList[i][0][1][0])+300, int(trackList[i][0][1][1])+300]
        p3 = [int(trackList[i][0][2][0])+300, int(trackList[i][0][2][1])+300]
        p4 = [int(trackList[i][0][3][0])+300, int(trackList[i][0][3][1])+300]

        img[p1[0],p1[1]]=255
        img[p2[0], p1[1]] = 255
        img[p3[0], p1[1]] = 255
        img[p4[0], p1[1]] = 255

    # cv2.imshow("track",img)
    im = Image.fromarray(img)
    out=im.convert("L")
    out.save("./track_simple.png")



def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def process_state_image2(state):

    fakeState = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    fakeState = fakeState.astype(float)
    fakeState /= 255.0
    # fakeState=state
    total=0
    for i in range(96):
        for j in range(96):
            total=total+fakeState[i][j]
    return fakeState,total/(96*96)



def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))







if __name__ == '__main__':

    print(1==1 and 2==3 and 3==3)
    print("111")
