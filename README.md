# Audio model for Speech/Music/SFX discrimination based on VGGish and Modelling of Experts' Decisions on assigning importances on radio objects

## Introduction 

This model is provided as supplementary material for the following paper:

```
@inproceedings{choudakisward2019dafx,
  title={MODELLING EXPERTS' DECISIONS ON ASSIGNING NARRATIVE IMPORTANCES OF
OBJECTS IN A RADIO DRAMA MIX},
  author={Chourdakis, E. T. and Ward, L. and Paradis, M. and Reiss, J.D. },
  year={2019},
  booktitle={Int. Conf. Digital Audio Effects},
  pages={[Under Review]},
}

```

## Requirements

The model has been found to be working with the following versions


| Module         | Version   |
| -------------- | --------- |
| `python3`      | 3.6.7     |
| `tensorflow`   | 1.12.0    |
| `keras`        | 2.2.4     |

There are also several other requirements listed in `requirements.txt`

## Installation 
 
Please run the following:

```
git clone git@github.com:bbc/audio-dafx2019-automatic.git

cd audio-dafx2019-automatic
```
Download the `.h5` and `.json` files from [v004](https://github.com/bbc/audio-dafx2019-automatic/releases/tag/v004) to the same directory.
```
# (optional) 
virtualenv venv
source venv/bin/activate
# (/optional)

pip3 install -r requirements.txt
```

You also need to download [vggish_input.py](https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/vggish_input.py), [mel_features.py](https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/mel_features.py), and [vggish_params.py](https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/vggish_params.py) put it into the same directory.

```
curl -O https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/vggish_input.py
curl -O https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/mel_features.py
curl -O https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/vggish_params.py
```

and make sure you read and agree with their LICENSES

```
head -n 14 vggish_input.py
head -n 14 mel_features.py
head -n 14 vggish_params.py
```

## Classifying audio to speech/music/SFX

Suppose you have a bunch of track files you need to classify to either speech/music or sound effects. First, you need to make sure that they are mono tracks 22050Hz/16bit (we recommend using [SoX](http://sox.sourceforge.net/)):

```
sox  input.wav -b 16 output.wav rate 22050 remix 1-2
```

Suppose your files are in a path `audio/`, you can classify the files in that folder by running:

```
python3 classify.py --model music_speech_sfx_discriminator audio/
```

You will then have a file ```output.csv``` with the results:

```
cat output.csv
```


Happy classifying~~!

## Assigning importances to a directory of radio track stems

Suppose you have a folder of stereo stems (sampling rate does not matter) ```stems/``` containing `.wav` files. In order to assign an importance level ( 0 -- low importance, 3 -- essential importance) to each of those, download the models seen in the previous section (Installation) to the same folder as `assign.py` and then run:
```
python3 assign.py stems
```

This will generate a file `output.csv` with the list of filenames for each stem, as well features and assigned importances. For more options see

```
python3 assign.py --help
```

## Training the Music/Speech/SFx classification model

Please see [MusicSpeechSFxDiscrimination.ipynb](https://github.com/bbc/audio-dafx2019-automatic/blob/master/MusicSpeechSFxDiscrimination.ipynb)

## Contact

Questions or issues about the model should either be raised here or addressed to Emmanouil Theofanis Chourdakis <e.t.chourdakis__Aaa.t!__qmul.ac.uk>





