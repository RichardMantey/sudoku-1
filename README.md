## Charles and Richard's branch

# Can Convolutional Neural Networks Crack Sudoku Puzzles?

Sudoku is a popular number puzzle that requires you to fill blanks in a 9X9 grid with digits so that each column, each row, and each of the nine 3×3 subgrids contains all of the digits from 1 to 9. There have been various approaches to solving that, including computational ones. In this project, we show that simple convolutional neural networks have the potential to crack Sudoku without any rule-based postprocessing.

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow == 1.1
	
## Background
* To see what Sudoku is, check the [wikipedia](https://en.wikipedia.org/wiki/Sudoku)
* To investigate this task comprehensively, read through [McGuire et al. 2013](https://arxiv.org/pdf/1201.0749.pdf).

## Dataset
* 1M games were generated using `generate_sudoku.py` for training. I've uploaded them on the Kaggle dataset storage. They are available [here](https://www.kaggle.com/bryanpark/sudoku/downloads/sudoku.zip).
* 30 authentic games were collected from http://1sudoku.com.

## Model description
* 10 blocks of convolution layers of kernel size 3.

## File description
  * `generate_sudoku.py` create sudoku games. You don't have to run this. Instead, download [pre-generated games](https://www.kaggle.com/bryanpark/sudoku/downloads/sudoku.zip).
  * `hyperparams.py` includes all adjustable hyper parameters.
  * `data_load.py` loads data and put them in queues so multiple mini-bach data are generated in parallel.
  * `modules.py` contains some wrapper functions.
  * `train.py` is for training.
  * `test.py` is for test.
  

## Training
* STEP 1. Download and extract [training data](https://www.kaggle.com/bryanpark/sudoku/downloads/sudoku.zip).
* STEP 2. Run `python train.py`. Or download the [pretrained file](https://www.dropbox.com/s/ipnwnorc7nz5hpe/logdir.tar.gz?dl=0).

## Test
* Run `python test.py`.

## Evaluation Metric

Accuracy is defined as 

Number of blanks where the prediction matched the solution / Number of blanks.


## Papers that referenced this repository

  * [OptNet: Differentiable Optimization as a Layer in Neural Networks](http://proceedings.mlr.press/v70/amos17a/amos17a.pdf)
  * [Recurrent Relational Networks for Complex Relational Reasoning](https://arxiv.org/abs/1711.08028)



