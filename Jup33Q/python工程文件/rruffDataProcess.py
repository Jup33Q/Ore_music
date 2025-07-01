import ramanspy
from ramanspy import Spectrum
import numpy as np
import pandas as pd

def loadDataFrame(NameOfDataSet:str='fair_oriented')->pd.DataFrame:
    data = ramanspy.datasets.rruff(NameOfDataSet)
    data_dict = dict()
    spectra, details = data
    for k in details[0].keys():
        data_dict[k] = [d[k] if k in d.keys() else '' for d in details]
    data_dict['Spectra'] = spectra
    df = pd.DataFrame(data_dict)
    df.drop('##END', axis=1, inplace=True)
    return df

def vectorisedBand(spectrum:Spectrum)->np.vectorize:
    return np.vectorize(spectrum.band)

def ReSamplingSpectrum(spectrum:Spectrum,rate:int=10000):
    x_min, x_max = spectrum.spectral_axis.min(), spectrum.spectral_axis.max()
    wavenumbers = np.linspace(x_min, x_max, rate)
    spectrum_data = vectorisedBand(spectrum)(wavenumbers)

    return Spectrum(spectrum_data,wavenumbers)

def getLogSpectrum(spectrum:Spectrum)->Spectrum:
    original_x_axis=spectrum.spectral_axis
    y=vectorisedBand(spectrum)(original_x_axis)
    return Spectrum(y,np.log(original_x_axis/spectrum.spectral_axis.mean()))

if __name__ == '__main__':
    df=loadDataFrame('fair_oriented')

    spectrum=df['Spectra'].iloc[0]
    spectrum=ReSamplingSpectrum(spectrum)

