#  Digital-Adaptive-Circuits-And-Learning-Systems

## Introduction
The dataset [Spoken Wikipedia Corpora](https://nats.gitlab.io/swc/) is saved in `./Dataset/English spoken wikipedia/english/`

## Prereqs (at least)
* [Python 3.9.5](https://www.python.org/) 
* [Matplotlib 3.4.1](https://pypi.org/project/matplotlib/)
* [Numpy 1.20.0](https://numpy.org/)
* [PySoundFile 0.9.0](https://pypi.org/project/PySoundFile/)
* [Librosa 0.8.1](https://librosa.org/doc/latest/index.html)
* [PyTorch 1.9.0](https://pytorch.org/)

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

* Install pytorch

	`pip3 install torch torchvision torchaudio`
	
2. Run
	
	2.1.a If you don't have the training and validation features please download them from the following link:
	
	* [Training_validation_features.rar](https://drive.google.com/file/d/1Eta9WrOY2Acq1-upCGOVrg3etVprvqAO/view?usp=sharing)
	
	Next unzip them.
	
	2.1.b. If don't want to download the training and validation features, you can create your own training and validation features. 
	Please open a command window and type in order:
    
        `python xml_parser_readers.py`
        `python find_target_words.py`
        `python words_per_reader.py`
        `python preprocessing.py`
		`python dataset_manager.py`
		
	2.2 Run the training script 
	
		`python training.py`
	
	2.3 Run the validation script
	
		`python validation.py`
	
## The code
[json_manager.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/json_manager.py) is used to read and write json files.

[xml_parser_readers.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/xml_parser_readers.py) creates [readers_paths.json](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/readers_paths.json).

```
[
    {
        "reader_name": "the epopt",
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

[dataset_manager.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/dataset_manager.py) creates the training, validation and test features. They will be saved in `Training_features`, `Validation_features` and `Test_features` folders. Each of these folders has directories, named after the reader name, which contain the features saved in torch tensor format like `word_name.pt` (examples: `as.pt`, `and.pt`, `with.pt`, etc) with dimension 26 x 128 x 51.

[loss.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/loss.py) is used to calculate the loss for each episode in the prototypical network.

[model.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/model.py) is used to create the prototypical network model: four CNN blocks, each of which has a convolutional layer with a 3 x 3 kernel, a batch normalization layer, a ReLU activation layer, and a 2 x 2 maxpooling layer.

[training.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/training.py) is used for training the neural network.

Average training loss for each network:

| C, K   | Prototypical Network | Siamese Network | Matching Network | Relation Network |
|--------|----------------------|-----------------|------------------|------------------|
| 2, 1   | 0.38632819493522547  |                 |                  |                  |
| 2, 5   |                      |                 |                  |                  |
| 5, 1   |                      |                 |                  |                  |
| 5, 5   |                      |                 |                  |                  |
| 10, 1  |                      |                 |                  |                  |
| 10, 5  |                      |                 |                  |                  |
| 10, 10 |                      |                 |                  |                  |

Average training accuracy for each network:

| C, K   | Prototypical Network | Siamese Network | Matching Network | Relation Network |
|--------|----------------------|-----------------|------------------|------------------|
| 2, 1   | 0.844255729166666    |                 |                  |                  |
| 2, 5   |                      |                 |                  |                  |
| 5, 1   |                      |                 |                  |                  |
| 5, 5   |                      |                 |                  |                  |
| 10, 1  |                      |                 |                  |                  |
| 10, 5  |                      |                 |                  |                  |
| 10, 10 |                      |                 |                  |                  |

[validation.py](https://github.com/MatteoOrlandini/Digital-Adaptive-Circuits-And-Learning-Systems/blob/main/validation.py) is used for the validation of the neural network.

Average validation loss for each network:

| C, K   | Prototypical Network | Siamese Network | Matching Network | Relation Network |
|--------|----------------------|-----------------|------------------|------------------|
| 2, 1   | 0.37211547262442374  |                 |                  |                  |
| 2, 5   |                      |                 |                  |                  |
| 5, 1   |                      |                 |                  |                  |
| 5, 5   |                      |                 |                  |                  |
| 10, 1  |                      |                 |                  |                  |
| 10, 5  |                      |                 |                  |                  |
| 10, 10 |                      |                 |                  |                  |

Average validation accuracy for each network:

| C, K   | Prototypical Network | Siamese Network | Matching Network | Relation Network |
|--------|----------------------|-----------------|------------------|------------------|
| 2, 1   | 0.8423802083333334   |                 |                  |                  |
| 2, 5   |                      |                 |                  |                  |
| 5, 1   |                      |                 |                  |                  |
| 5, 5   |                      |                 |                  |                  |
| 10, 1  |                      |                 |                  |                  |
| 10, 5  |                      |                 |                  |                  |
| 10, 10 |                      |                 |                  |                  |
