#!/bin/python

# Model for classifying 
import librosa
import librosa.feature
import numpy as np
import argparse
import os
import glob
import vggish_input
import pandas as pd


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Calculates probabilities for a given sound file")
    parser.add_argument('path', help='filename or directory path of files to compute probabilities, must be readable by librosa')
    parser.add_argument('--sampleRate', help='sampleRate the model was trained with', type=float, default=22050.0)
    parser.add_argument('--model', help='name of the model to work with, must be in the same directory', type=str,  default='music_speech_sfx_discriminator_vggish')
    parser.add_argument('--output', help='output csv file of two columns, one with the filename, the other with wether the file is speech/music or sfx', type=str, default='output.csv')
    
    args = parser.parse_args()

    path = args.path

    if not os.path.isdir(path):
        filenames = [path]
    else:
        filenames = glob.glob(os.path.join(path, '*.wav')) + \
                    glob.glob(os.path.join(path, '*.mp3')) + \
                    glob.glob(os.path.join(path, '*.flac'))

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

    pl_output = "% Types of files "
    
    records = []
    
    for input_file in filenames:
        print(input_file)
        y, sr = librosa.load(input_file, args.sampleRate)
        S, phase = librosa.magphase(librosa.stft(y))
        rms = librosa.feature.rmse(S=S)
        segment_borders = np.diff((rms.T >= 0.001).astype(np.float).flatten()) # Under -60dbFS

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
            segments.append((left_border, len(y)))

        # Collate the active segments
        active_y = np.hstack([y[segment[0]:segment[1]] for segment in segments])

        input_batch = vggish_input.waveform_to_examples(active_y, sr)
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
        
        class_ = ['music', 'speech', 'sfx'][np.argmax([p_music, p_speech, p_sfx])]
        
        records.append({ 
            'asset_fname': basename,
            'p_music': p_music,
            'p_speech' : p_speech,
            'p_sfx' : p_sfx,
            'class' : class_
        })

        pl_output += '{1}::type("{0}",music); {2}::type("{0}",speech); {3}::type("{0}",sfx).\n'.format(basename, p_music, p_speech, p_sfx)
    

    with open('out.pl', 'w') as f:
        f.write(pl_output+'\n')
        
    df = pd.DataFrame.from_records(records)
    df.to_csv(args.output)
