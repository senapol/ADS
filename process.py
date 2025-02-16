import pandas as pd
import numpy as np
import plotly.express as px

def process_milex_excel(path):
    'Read and convert Milex xlsx file to pandas df'

    df = pd.read_excel(path, sheet_name='Constant (2022) US$', skiprows=5)

    