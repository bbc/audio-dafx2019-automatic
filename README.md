# Audio model for Speech/Music/SFX discrimination based on VGGish

## Introduction 

This model is provided as supplementary material for the following paper:

```
@article{choudakisward2019dafx,
  title={Modelling Human Experts' Narrative Importance Assignments in a Radio Drama Using Mixture Models},
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

curl -O https://github.com/bbc/audio-dafx2019-automatic/releases/download/disc001/music_speech_sfx_discriminator_vggish_gtzan.pkl

(optional) 
virtualenv venv
source venv/bin/activate
(/optional)

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

Suppose you have a bunch of track files you need to classify to either speech/music or sound effects. First, you need to make sure that they are stereo tracks sampled at 22050Hz (we recommend using [SoX](http://sox.sourceforge.net/)) for resampling:

```
sox input.wav -ar 22050 output.wav
```

Suppose your files are in a path `audio/`, you can classify the files in that folder by running:

```
python3 classify.py music_speech_sfx_discriminator_vggish_gtzan audio/
```

You will then have a file ```output.csv``` with the results:

```
cat output.csv
```


Happy classifying~~!

## Contact

Questions or issues about the model should either be raised here or addressed to Emmanouil Theofanis Chourdakis <e.t.chourdakis__Aaa.t!__qmul.ac.uk>





