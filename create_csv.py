import pandas as pd 
import os 
from pathlib import Path 

path = "/home/nero/AI4VN/sample_data/train"

csv_file = "/home/nero/AI4VN/AI4VN-Hackathon/train.csv"

df = pd.DataFrame(columns = ['filename', 'label']) 
for i, label in enumerate(sorted(Path(path).iterdir())):
    for file in sorted(label.iterdir()):
        df = df.append({'filename': file, 'label': i}, ignore_index =True)

df.to_csv(csv_file, index = False)
