from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
# Define the path to your downloaded and extracted NASA mat files
# Example: 'milling_wear_dataset/'
#dataset_dir = "path_to_your_extracted_folder/"


SENSOR_COLUMNS = [
    'smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle'
]


def load_mill_dataframe(path='mill.mat'):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    runs = np.atleast_1d(mat['mill']).ravel()
    rows = []
    for record in runs:
        row = {name: getattr(record, name) for name in record._fieldnames}
        for name in SENSOR_COLUMNS:
            row[name] = np.asarray(row[name]).ravel()
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = load_mill_dataframe()
    print(f'DataFrame shape: {df.shape}')
    print(df[['case', 'run', 'VB', 'time', 'DOC', 'feed', 'material']].head())
    print('First vibration signal shape:', df.loc[0, 'vib_spindle'].shape)
    # Example: extract simple features for a conventional ML table
    # df['vib_spindle_rms'] = df['vib_spindle'].apply(
    #     lambda x: np.sqrt(np.mean(x ** 2))
    # )
