#  Digital-Adaptive-Circuits-And-Learning-Systems

## Introduction
The dataset [Spoken Wikipedia Corpora](https://nats.gitlab.io/swc/) is saved in `./Dataset/English spoken wikipedia/english/`

## Prereqs (at least)
* [Python 3.9.5](https://www.python.org/) 
* [Matplotlib 3.4.1](https://pypi.org/project/matplotlib/)
* [Numpy 1.20.0](https://numpy.org/)
* [PySoundFile 0.9.0](https://pypi.org/project/PySoundFile/)
* [Librosa 0.8.1](https://librosa.org/doc/latest/index.html)

## How to run 
1. Install the libraries
* matplotlib

Open a command window and type:

`python -m pip install -U pip`

`python -m pip install -U matplotlib`

* Install numpy

    `pip install numpy`

* Install PySoundFile

    `pip install PySoundFile`

* Install librosa

    `pip install librosa`

2. Run

    Open a command window and type in order:
    
        `python xml_parser_readers.py`
        `python find_target_words.py`
        `python words_per_reader.py`

## The code
[xml_parser_readers.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/xml_parser_readers.py) creates [readers_paths.json](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/readers_paths.json).

```
[
    {
        "reader_name": "the epopt",
        "frequency": 5,
        "folder": [
            "(I_Can%27t_Get_No)_Satisfaction",
            "Ceremonial_ship_launching",
            "Limerence",
            "Revolt_of_the_Admirals",
            "Ship_commissioning"
        ]
    },
    {
        "reader_name": "wodup",
        "frequency": 5,
        "folder": [
            "0.999..%2e",
            "Execution_by_elephant",
            "Hell_Is_Other_Robots",
            "Tom_Bosley",
            "Truthiness"
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
    {
        "word": "the",
        "frequency": 75,
        "start": [
            4160,
            49930,
            53680,
            63160,
            68070,
            69060,
            80550,
            91410,
            92060,
            95140,
            100510,
            105720,
            107100,
            110370,
            113520,
            115380,
            153410,
            154840,
            161450,
            165500,
            168200,
            169250,
            173940,
            175440,
            181980,
            182500,
            187640,
            190500,
            193660,
            202040,
            203040,
            205350,
            219690,
            222400,
            254330,
            260150,
            261150,
            268730,
            293250,
            301550,
            302840,
            307260,
            308460,
            323660,
            337210,
            373170,
            383790,
            420810,
            425200,
            427440,
            443990,
            449740,
            471220,
            475730,
            489030,
            489910,
            494210,
            500170,
            502590,
            506180,
            533640,
            536690,
            537970,
            539050,
            556290,
            565200,
            567270,
            570770,
            584690,
            591890,
            593170,
            604600,
            611510,
            612810,
            616480
        ],
        "end": [
            4320,
            50030,
            53710,
            63260,
            68170,
            69170,
            80660,
            91500,
            92170,
            95250,
            100600,
            105840,
            107210,
            110460,
            113730,
            115670,
            153520,
            154970,
            161560,
            165590,
            168300,
            169550,
            174270,
            175520,
            182080,
            182600,
            187740,
            190600,
            193770,
            202160,
            203390,
            205450,
            219820,
            222490,
            254450,
            260210,
            261290,
            268830,
            293350,
            301690,
            302940,
            307390,
            308570,
            323770,
            337280,
            373620,
            383920,
            420920,
            425270,
            427520,
            444110,
            449830,
            471360,
            475930,
            489140,
            490030,
            494310,
            500300,
            502730,
            506320,
            533750,
            536800,
            538130,
            539130,
            556430,
            565320,
            567360,
            570850,
            584780,
            592400,
            593500,
            604690,
            612030,
            613320,
            616760
        ]
    },
    .
    .
    .
]
```

[words_per_reader.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/mel_spectrogram.py) creates a 
```readers_words.json```

```
[
    {
        "reader_name": "the epopt",
        "words": [
            {
                "word": "i",
                "folders": [
                    {
                        "folder": "(I_Can%27t_Get_No)_Satisfaction",
                        "start": [
                            [
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
                            ]
                        ],
                        "end": [
                            [
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
                        ]
                    }
                ]
            },
            {
                "word": "the",
                "folders": [
                    {
                        "folder": "(I_Can%27t_Get_No)_Satisfaction",
                        "start": [
                            [
                                4160,
                                49930,
                                53680,
                                63160,
                                68070,
                                69060,
                                80550,
                                91410,
                                92060,
                                95140,
                                100510,
                                105720,
                                107100,
                                110370,
                                113520,
                                115380,
                                153410,
                                154840,
                                161450,
                                165500,
                                168200,
                                169250,
                                173940,
                                175440,
                                181980,
                                182500,
                                187640,
                                190500,
                                193660,
                                202040,
                                203040,
                                205350,
                                219690,
                                222400,
                                254330,
                                260150,
                                261150,
                                268730,
                                293250,
                                301550,
                                302840,
                                307260,
                                308460,
                                323660,
                                337210,
                                373170,
                                383790,
                                420810,
                                425200,
                                427440,
                                443990,
                                449740,
                                471220,
                                475730,
                                489030,
                                489910,
                                494210,
                                500170,
                                502590,
                                506180,
                                533640,
                                536690,
                                537970,
                                539050,
                                556290,
                                565200,
                                567270,
                                570770,
                                584690,
                                591890,
                                593170,
                                604600,
                                611510,
                                612810,
                                616480
                            ]
                        ],
                        "end": [
                            [
                                4320,
                                50030,
                                53710,
                                63260,
                                68170,
                                69170,
                                80660,
                                91500,
                                92170,
                                95250,
                                100600,
                                105840,
                                107210,
                                110460,
                                113730,
                                115670,
                                153520,
                                154970,
                                161560,
                                165590,
                                168300,
                                169550,
                                174270,
                                175520,
                                182080,
                                182600,
                                187740,
                                190600,
                                193770,
                                202160,
                                203390,
                                205450,
                                219820,
                                222490,
                                254450,
                                260210,
                                261290,
                                268830,
                                293350,
                                301690,
                                302940,
                                307390,
                                308570,
                                323770,
                                337280,
                                373620,
                                383920,
                                420920,
                                425270,
                                427520,
                                444110,
                                449830,
                                471360,
                                475930,
                                489140,
                                490030,
                                494310,
                                500300,
                                502730,
                                506320,
                                533750,
                                536800,
                                538130,
                                539130,
                                556430,
                                565320,
                                567360,
                                570850,
                                584780,
                                592400,
                                593500,
                                604690,
                                612030,
                                613320,
                                616760
                            ]
                        ]
                    },
                    .
                    .
                    .
```

[mel_spectrogram.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/mel_spectrogram.py) computes the 128 bin log-mel spectrogram.
