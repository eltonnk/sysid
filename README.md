# How to use this App

## Setup Python Virtual Environnement

### On Windows

Open a terminal in the current repo. 

Enter the following command:

```
py -m venv env

```

In the same terminal, enter the following command:

```
.\env\Scripts\activate.bat 
```

You have now activated your python virtual environnement in this terminal. 

Now go to https://www.lfd.uci.edu/~gohlke/pythonlibs/ and download the numpy+mkl wheel, as well as the slycot wheel.
the file names should look something like this:

```
numpy-1.22.4+mkl-cp310-cp310-win_amd64.whl
SciPy-1.8.1-cp310-cp310-win_amd64.whl
slycot-0.4.0-cp310-cp310-win_amd64.whl
pandas-1.4.3-cp310-cp310-win_amd64.whl
```

Dowload the files that correspond to the python version on your system.

Put both these files in the custom_wheels folder in this repo.

In your requirements.txt file in this repo, copy the file names of the wheel files you just downloaded. requirements.txt should look like this:

```
custom_wheels/numpy-1.22.4+mkl-cp310-cp310-win_amd64.whl
custom_wheels/SciPy-1.8.1-cp310-cp310-win_amd64.whl
control
custom_wheels/slycot-0.4.0-cp310-cp310-win_amd64.whl
scikit-learn
dataclasses-json
custom_wheels/pandas-1.4.3-cp310-cp310-win_amd64.whl
```

> **Note**
> Please keep the packages in the same order. It is very important that numpy+mkl is the first package installed.

Now enter the following command in your terminal, with your virtual environnement activated:

```
pip install -r requirements.txt
```

### On Linux

# TODO 

# How to store data and organise files

When using part1 or part2 files, need to select in which folder are or will be inserted input/output files. This folder needs to contain

- main_folder (folder where all input/output files are to be found, can have any name)
    - DATA (where all data will be found)
        - reference/ref.csv (this is used to compare performance of closed loop plant vs reference)
        - data_file_0001.csv
        - data_file_0002.csv
        - etc
        
    - good_plants
        - v1 (at least)
            - good_plant_v1.txt
        - v2
        - v3
        - etc
    - try_all_results
    - training_plans
        - plan.json

## File Description

### data_file_00XX.csv

    only needs to be in DATA and end with .csv
    should have
    t,r,u,y
    columns (or more, those wont be used)
    with column headers

    if using other column names, modify the "sensor_data_column_names" section in plan.json
    file explained below

    should not change name or add other data files in DATA if you want the
    good_plants_vX.txt file to train exactly the plants it is meant to train
    (this will be changed in the future)

### plan.json
    - sensor_data_column_names
    
    Used to determine which columns in csv data files used for training and testing correspond to time ("t"), command ("r"), input ("u") and output ("y") data of plant we want to characterize

    - plants_to_train

    tells numerator and denominator order of plants to train, also wheter to use normalization or standardization while training


    format:

    ```json
    {
        "sensor_data_column_names":
        {
            "t": "Time (s)",
            "r": "Command (Nm)",
            "u": "Motor Voltage (V)",
            "y": "Torque Output (Nm)"
        },
        "plants_to_train":
        [
            {
                "num_order": 5,
                "denum_order": 6,
                "better_cond_method" : "normalizing"
            },
            {
                "num_order": 2,
                "denum_order": 3,
                "better_cond_method" : "standardizing"
            }
        ]
    }
    ```

> **Warning**
> Do not train on more than 4-6 files, and each file should be no more than 500-600 kb, or else your computer will crash. Respecting the max number of files is more important than respecting the file size, but both are important as else this program will fill up your memory.
