import pandas as pd
from tkinter.filedialog import askopenfilename
import numpy as np

MAIN_FILE_NAME = askopenfilename(title='Select data File')

df =pd.read_csv(MAIN_FILE_NAME)

t = np.array(df["t"])

print(t)