

## Environment:

see requirements.txt

```shell
pip install -r requirements.txt

```


to install the origin gym and racing cars, use:


```shell
pip install gym
conda install box2d

```

to install the editable gym and racing cars, coming soon...


## Data Preparation


```shell
python data_colector.py --path=="yourPath" --epoch=="number of runing epoch, default is 50"

```

path is where the generated data will save, and epoch is the iterations. (On average, each epoch will generate about 100 frames)



1. add the modified gym or box2d (there's version conflict between these 2, since we use racing-car-v0 instead of v2)

There're 2 modules need to modify to customize some parameters like track width, grass color, ect..


2. write a guide to run the code in README.md

