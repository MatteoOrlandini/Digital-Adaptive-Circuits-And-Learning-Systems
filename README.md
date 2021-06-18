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
        `python preprocessing.py`

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
            .
            .
            .
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
					}
				]
			},
			{
				"word": "the",
				"folders": [
					{
						"folder": "(I_Can%27t_Get_No)_Satisfaction",
						"start": [
							4160,
                            .
                            .
                            .
```

[preprocessing.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/preprocessing.py) creates `training_readers.json`, `test_readers.json` and `validation_readers.json`.

```
[
    {
        "reader_name": "tonyle",
        "words": [
            {
                "word": "three",
                "start": [
                    460,
                    9960,
                    70660,
                    194030,
                    361910,
                    458660,
                    462930,
                    520510,
                    578400,
                    706680,
                    757710,
                    782040,
                    800340,
                    824390,
                    843790,
                    858210,
                    871590,
                    887460,
                    919370,
                    943290,
                    958850,
                    967830,
                    977170,
                    988750,
                    1021640,
                    1031930,
                    1051520,
                    1059350,
                    1087490,
                    1137170,
                    1153190,
                    1212100,
                    1217700,
                    1251470,
                    1281600,
                    1312910,
                    1330150,
                    1343530,
                    1370660,
                    1604690,
                    1677780,
                    1714340,
                    1869740,
                    1895580,
                    1907390,
                    1931410,
                    1937330,
                    1941880,
                    1961660,
                    889860,
                    905430,
                    1075130,
                    1266220,
                    1268730,
                    1274300,
                    1301000,
                    1305590,
                    1308130,
                    1316960,
                    1621930,
                    1877550,
                    1948730,
                    2281470
                ],
                "end": [
                    740,
                    10270,
                    71140,
                    194320,
                    362140,
                    458910,
                    463150,
                    520770,
                    578790,
                    706980,
                    758090,
                    782260,
                    800540,
                    824660,
                    844030,
                    858520,
                    871830,
                    887670,
                    919590,
                    943530,
                    959190,
                    968080,
                    977400,
                    988980,
                    1021970,
                    1032190,
                    1051730,
                    1059560,
                    1087750,
                    1137400,
                    1153460,
                    1212320,
                    1217970,
                    1251700,
                    1281880,
                    1313170,
                    1330420,
                    1343740,
                    1370870,
                    1604950,
                    1678030,
                    1714580,
                    1870000,
                    1895900,
                    1907630,
                    1931640,
                    1937560,
                    1942700,
                    1961930,
                    890360,
                    905730,
                    1075410,
                    1266550,
                    1269020,
                    1274600,
                    1301540,
                    1306140,
                    1308400,
                    1317210,
                    1622210,
                    1878020,
                    1949320,
                    2281980
                ],
                "folder": [
                    "300_(film)",
                    "300_(film)",
                    "300_(film)",
                    "300_(film)",
                    .
                    .
                    .
```

[mel_spectrogram.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/mel_spectrogram.py) computes the 128 bin log-mel spectrogram to a 0.5 second window centered in the middle of the word.
