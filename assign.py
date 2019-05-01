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

hopSize = 512
frameSize = 2048
sampleRate = 22050

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts features and assigns importances to .wav tracks in a directory")
    parser.add_argument('path', help='filename or directory path of .wav files to assign importances')
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
    
    df = pd.DataFrame.from_records(records)
    df.to_csv(args.output)
    