# How to use this App

## Setup Python Virtual Environnement

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
```

> **Note**
> Please keep the packages in the same order. It is very important that numpy+mkl is the first package installed.

Now enter the following command in your terminal, with your virtual environnement activated:

```
pip install -r requirements.txt
```

# Warning

Do not train on more than 4-6 files, and each file should be no more than 500-600 kb, or else your computer will crash.
Respecting the max number of files is more important than respecting the file size, but both are important as else this
program will fill up your ram