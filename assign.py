#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:19:12 2019

@author: Emmanouil Theofanis Chourdakis <e.t.chourdakis@qmul.ac.uk>

Utility that extracts information needed for automatic assignment of
narrative importances. More specifically, the information extracted is:
    
    1. How much of the file is music, or speech.
    2. The total duration of the file.
    3. The True Peak-to-Relative loudness ratio
    
It then loads uses a mixture model to assign importances, saves the results
to a .csv file.
"""

import argparse
import librosa
import librosa.feature
import numpy as np
import argparse
import essentia
import essentia.standard
import os
import glob
import vggish_input
import pandas as pd
import tqdm
import pyloudness as ld
import pyro
from pyro import poutine
from pyro.infer import infer_discrete
import joblib
import argparse
import torch

# Functions for the mixture model
def model_b(data):
    
    importances = pyro.sample("importances", pyro.distributions.Dirichlet(2*torch.ones(4)))
    with pyro.plate('components', 4) as ind:
    
        rms_alphas = pyro.sample('rms_alphas', pyro.distributions.Gamma(concentration=7.5*torch.ones(4), rate=torch.ones(4)))
        rms_betas = pyro.sample('rms_betas', pyro.distributions.Gamma(concentration=7.5*torch.ones(4), rate=torch.ones(4)))
        
        pspeech_alphas = pyro.sample('pspeech_alphas', pyro.distributions.Gamma(concentration=7.5*torch.ones(4), rate=torch.ones(4))) 
        pspeech_betas = pyro.sample('pspeech_betas', pyro.distributions.Gamma(concentration=7.5*torch.ones(4), rate=torch.ones(4)))
        
        pmusic_alphas = pyro.sample('pmusic_alphas', pyro.distributions.Gamma(concentration=7.5*torch.ones(4), rate=torch.ones(4))) 
        pmusic_betas = pyro.sample('pmusic_betas', pyro.distributions.Gamma(concentration=7.5*torch.ones(4), rate=torch.ones(4)))

        dur_alphas = pyro.sample('dur_alphas', pyro.distributions.Gamma(concentration=7.5*torch.ones(4), rate=torch.ones(4))) 
        dur_betas = pyro.sample('dur_betas', pyro.distributions.Gamma(concentration=7.5*torch.ones(4), rate=torch.ones(4)))
                
    with pyro.plate('observations', data.size()[1]) as ind:
      
         assigned = pyro.sample("assigned", pyro.distributions.Categorical(importances), infer={"enumerate": "parallel"}).long()
         rms = pyro.sample("rms", pyro.distributions.Beta(rms_alphas[assigned], rms_betas[assigned]), obs=data[1,ind], infer={"enumerate": "parallel"})
         pspeech = pyro.sample("pspeech", pyro.distributions.Beta(pspeech_alphas[assigned],pspeech_betas[assigned]), 
                               obs=data[2,ind], infer={"enumerate": "parallel"})
         pmusic = pyro.sample("pmusic", pyro.distributions.Beta(pmusic_alphas[assigned],pmusic_betas[assigned]), 
                               obs=data[3,ind], infer={"enumerate": "parallel"})
         dur = pyro.sample("dur", pyro.distributions.Gamma(concentration=dur_alphas[assigned], rate=dur_betas[assigned]), obs=data[4,ind], infer={"enumerate": "parallel"})
        
    
    return assigned, rms 

def classifier(data, temperature=0):
    inferred_model = infer_discrete(trained_model, temperature=temperature,
                                    first_available_dim=-2)  # avoid conflict with data plate
    trace = poutine.trace(inferred_model).get_trace(data)
    return trace.nodes["assigned"]["value"]    

# Constants for feature extraction
hopSize = 512
frameSize = 2048
sampleRate = 22050

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts features and assigns importances to .wav tracks in a directory")
    parser.add_argument('path', help='filename or directory path of .wav files to assign importances')
    parser.add_argument('--joblib', help='joblib file of the guide trace ', type=str, default='guide_trace.joblib')    
    parser.add_argument('--model', help='name of the speech/music/sfx model to work with, must be in the same directory', type=str,  default='music_speech_sfx_discriminator_vggish')
    parser.add_argument('--output', help='output csv file of two columns, one with the filename, the other with its assigned importance', type=str, default='output.csv')
    
    args = parser.parse_args()

    path = args.path

    if not os.path.isdir(path):
        filenames = [path]
    else:
        filenames = glob.glob(os.path.join(path, '*.wav'))
    
    hop_size = 512

    # Load model
    from keras.models import model_from_json
    with open("{}.json".format(args.model)) as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights("{}.h5".format(args.model))

    print("Using model: {}".format(args.model))
    print("Architecture:")
    print(model.summary())

    records = []
    for input_file in tqdm.tqdm(filenames):
        # For duration extraction (using librosa and essentia at the same time seems)
        # overkill however we need to deal with it at the current version
        audio = essentia.standard.MonoLoader(filename = input_file, sampleRate=22050)()        
        
        # Loudness extractors 
        loudness_dict =  ld.get_loudness(input_file)
        true_peak = loudness_dict['True Peak']['Peak']
        integrated_loudness = loudness_dict['Integrated Loudness']['I']        
        
        # Music/Speech/Sfx discrimination
        frameGenerator = essentia.standard.FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize, startFromZero=True)
        w = essentia.standard.Windowing(size=frameSize)
        rmsx = essentia.standard.RMS()
    
        rms = []
        for frame in frameGenerator:
            rms.append(rmsx(frame))
    
        rms = np.array(rms)
    
        segment_borders = np.diff((rms >= 0.001*np.max(rms)).astype(np.float).flatten()) # When under -60dbFS    
        segments = []
        left_border = 0
        right_border = 0
        
        for n in range(len(segment_borders)):
            if segment_borders[n] > 0:
                left_border = n*hop_size
            if segment_borders[n] < 0:
                right_border = n*hop_size
                segments.append((left_border, right_border))

        # If the last segment terminates after the end of file
        if left_border > right_border or left_border == 0 and right_border == 0:
            segments.append((left_border, len(audio)))
            
        # Slice audio according to segments
        slicer = essentia.standard.Slicer(startTimes=[s[0] for s in segments], 
                                          endTimes=[s[1] for s in segments],
                                          timeUnits='samples', sampleRate=22050)
        
        sliced_audio = np.hstack(slicer(audio))
        
        sliced_duration = essentia.standard.Duration(sampleRate=22050)(sliced_audio)            

        # Collate the active segments
        active_y = np.hstack([audio[segment[0]:segment[1]] for segment in segments])

        input_batch = vggish_input.waveform_to_examples(active_y, sampleRate)
        predictions = np.argmax(model.predict(input_batch[:,:,:,None]), axis=1)
        
        
        p_music = 0
        p_speech = 0
        p_sfx = 0
        sum_ = len(predictions)
        for n in range(len(predictions)):
            if predictions[n] == 0:
                p_music += 1/sum_
            elif predictions[n] == 1:
                p_speech += 1/sum_
            elif predictions[n] == 2:
                p_sfx += 1/sum_

        basename = os.path.basename(input_file)
        
        records.append({ 
            'asset_fname': basename,
            'p_music': p_music,
            'p_speech' : p_speech,
            'tpti' : true_peak/integrated_loudness,
            'total_duration': sliced_duration,
        })
    
    # Create initial dataframe with the features 
    df = pd.DataFrame.from_records(records)
    
    # Load the guide trace
    guide_trace = joblib.load(args.joblib)
    trained_model = poutine.replay(model_b, trace=guide_trace) # see notebook for this line
    
    # We are trying to decide an importance assignment value from the data given 4 features.
    # Our feature matrix (values) is therefore Nx4. The model (and therefore the classifier)
    # expects a Nx5 data matrix so we will add dummy values as the first column. Also convert it to
    # tensor.
    
    values = df[['tpti','p_speech','p_music','total_duration']].to_numpy()
    dummy_assignments = np.zeros((values.shape[0],1))
    data = np.hstack([dummy_assignments, values]).T
    
    # Apply some noise (like we did in the training)
    data[1,:] = data[1,:]*0.8 + 0.1 + np.random.randn(*data[1,:].shape)*0.001
    data[2,:] = data[2,:]*0.8 + 0.1 + np.random.randn(*data[2,:].shape)*0.001
    data[3,:] = data[3,:]*0.8 + 0.1 + np.random.randn(*data[3,:].shape)*0.001

    # Convert ot tensor
    data = torch.tensor(data).float()
    
    # Predict assignments
    assign_pred = classifier(data)
    df['assigned'] = assign_pred
    print(df[['asset_fname','assigned']])
    df.to_csv(args.output)    