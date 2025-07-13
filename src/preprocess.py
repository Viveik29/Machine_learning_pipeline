import sys
import pandas as pd
import numpy as np
import os
import yaml

params = yaml.safe_load(open('params.yaml'))['preprocess']
def preprocessor(input_path,output_path):
    data=pd.read_csv(input_path)
    print(data)
    print(data.shape)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    data.to_csv(output_path)
    print(f"preprocessed data saved at {output_path}")


if __name__=="__main__":
    preprocessor(params['input'],params['output'])
