import  numpy as np

if __name__ == '__main__':
    a = np.load("datas/safe.npy")
    b = np.load("datas/label.npy")
    count0=0
    count1=0
    countm=0
    for i in range(len(b)):
        if b[i]==0:
            count0=count0+1
        if b[i]==1:
            count1=count1+1
        if b[i]==-1:
            countm=countm+1

    print(count1/len(b))
    print("end")