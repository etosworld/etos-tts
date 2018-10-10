
### ETOS TTS

ETOS TTS,  aims to build a neural text-to-speech (TTS) that is able to transform text to speech in voices that are sampled in the wild.  It is a PyTorch Implementation of [Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model](https://arxiv.org/abs/1703.10135). 

### ETOS

- Based on ChromiumOS, ETOS is a full customed BlockChain OS. https://etos.world
- With ETOS Coin POW CPU Mining, ETOS Store BlockChain APPs and Games, and ETOS MLKit API inside.

### Usage


#### Requirements

- python 3.6 or later 
- pytorch 0.4 is tested

you can use pip to install other requirements.

```pip3 install -r requirements.txt```

#### Testing 

you can use pretrained model under ```models/may22``` and run the tts web server:

```python server.py -c server_conf.json```

Then go to ```http://127.0.0.1:8000``` and enjoy.


#### Data
Currently TTS provides data loaders for
- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)

### Training the network
To run your own training, you need to define a ```config.json``` file (simple template below) and call with the command.

```train.py --config_path config.json```

If you like to use specific set of GPUs.

```CUDA_VISIBLE_DEVICES="0,1,4" train.py --config_path config.json```

Each run creates an experiment folder with the corresponfing date and time, under the folder you set in ```config.json```. And if there is no checkpoint yet under that folder, it is going to be removed when you press Ctrl+C.

You can also enjoy Tensorboard with couple of good training logs, if you point ```--logdir``` the experiment folder.

Example ```config.json```:
```
{
  "num_mels": 80,
  "num_freq": 1025,
  "sample_rate": 22050,
  "frame_length_ms": 50,
  "frame_shift_ms": 12.5,
  "preemphasis": 0.97,
  "min_level_db": -100,
  "ref_level_db": 20,
  "embedding_size": 256,
  "text_cleaner": "english_cleaners",

  "epochs": 200,
  "lr": 0.002,
  "warmup_steps": 4000,
  "batch_size": 32,
  "eval_batch_size":32,
  "r": 5,
  "mk": 0.0,  // guidede attention loss weight. if 0 no use
  "priority_freq": true,  // freq range emphasis

  "griffin_lim_iters": 60,
  "power": 1.2,

  "dataset": "TWEB",
  "meta_file_train": "transcript_train.txt",
  "meta_file_val": "transcript_val.txt",
  "data_path": "/data/shared/BibleSpeech/",
  "min_seq_len": 0,
  "num_loader_workers": 8,

  "checkpoint": true,  // if save checkpoint per save_step
  "save_step": 200,
  "output_path": "/path/to/my_experiment",
}
```

#### TODO
- wavenet vocoder for better quality
- IAF or NAF for real time performance

#### References
- [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435.pdf)
- [Attention-Based models for speech recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf)
- [Char2Wav: End-to-End Speech Synthesis](https://openreview.net/pdf?id=B1VWyySKx)
- [VoiceLoop: Voice Fitting and Synthesis via a Phonological Loop](https://arxiv.org/pdf/1707.06588.pdf)
- [WaveRNN](https://arxiv.org/pdf/1802.08435.pdf)
- [Faster WaveNet](https://arxiv.org/abs/1611.09482)
- [Parallel WaveNet](https://arxiv.org/abs/1711.10433)
- [NAF](https://arxiv.org/abs/1804.00779)

#### Thanks
- https://github.com/keithito/tacotron
- https://github.com/r9y9/tacotron_pytorch
- https://github.com/mozilla/TTS

