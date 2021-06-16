#  Digital-Adaptive-Circuits-And-Learning-Systems

## Introduction


## Prereqs (at least)
* [Python 3.9.5](https://www.python.org/) 
* [Matplotlib 3.4.1](https://pypi.org/project/matplotlib/)
* [Numpy 1.20.0](https://numpy.org/)
* [PySoundFile 0.9.0](https://pypi.org/project/PySoundFile/)
* [Librosa 0.8.1](https://librosa.org/doc/latest/index.html)

## How to run 
1. Install matplotlib

Open a command window and type:

`python -m pip install -U pip`

`python -m pip install -U matplotlib`

Install numpy

`pip install numpy`

Install PySoundFile

`pip install PySoundFile`

Install librosa

`pip install librosa`

2. Run

Open a command window and type `python main.py`. 

## The code
[xml_parser_readers.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/xml_parser_readers.py) creates a [readers_paths.json](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/readers_paths.json) and [readers_paths.txt](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/readers_paths.txt)

```
[
    {
        "reader_name": "the epopt",
        "frequency": 5,
        "paths": [
            "./Dataset/English spoken wikipedia/english/(I_Can%27t_Get_No)_Satisfaction",
            "./Dataset/English spoken wikipedia/english/Ceremonial_ship_launching",
            "./Dataset/English spoken wikipedia/english/Limerence",
            "./Dataset/English spoken wikipedia/english/Revolt_of_the_Admirals",
            "./Dataset/English spoken wikipedia/english/Ship_commissioning"
        ]
    },
    {
        "reader_name": "wodup",
        "frequency": 5,
        "paths": [
            "./Dataset/English spoken wikipedia/english/0.999..%2e",
            "./Dataset/English spoken wikipedia/english/Execution_by_elephant",
            "./Dataset/English spoken wikipedia/english/Hell_Is_Other_Robots",
            "./Dataset/English spoken wikipedia/english/Tom_Bosley",
            "./Dataset/English spoken wikipedia/english/Truthiness"
        ]
    },
    .
    .
    .
]
```

[find_target_words.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/find_target_words.py) creates a ```word_count.json``` for each audio.

```
[
    {
        "word": "i",
        "frequency": 12,
        "start": [
            660,
            8800,
            115050,
            116300,
            117910,
            228560,
            273900,
            497150,
            518270,
            534740,
            543420,
            589280
        ],
        "end": [
            870,
            8940,
            115240,
            116360,
            118020,
            228690,
            274000,
            497340,
            518520,
            534860,
            543700,
            589360
        ]
    },
    .
    .
    .
]
```

[mel_spectrogram.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/mel_spectrogram.py) computes the 128 bin log-mel spectrogram.