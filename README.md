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

Accuracy is defined as: Number of blanks where the prediction matched the solution / Number of blanks.

## Next Steps

// Improve CNN
 - epoch nums
 - filter size
 - regularization
 - dropout
 - more tainining data (medium, hard, expert, evil)

// Search Problem
 - Use prob distributions in heuristic search

// RNN (Bidirectional with LSTM cells)
 - list of tuples (blank_cell, num_hints) where num_hints = sum of hints in row,col,square
 - sort list by num_hints
 - feed in row, col, sqaure of cell with highest num_hints to RNN
 - take joint prob for cell as solution
 - update board with prediction
 - recompute list with newly predicted cell
 - choose new largest num_hints cell and continue
 - (build structure)

// Deep RL
 - very difficult
 - we have structure for it though

Visualization
expected board --> our board

// Input should now be probability distribution
 - lets make it all one-hot
 - back to 9 labels
 - 0 has equal probability for all labels
 
 bi - directional lstm
 go across the row, column, square as individual inputs to the lstm
 each pos in row x batch x num_entries

only do softmax at end

feed forward layer to translate the outputs of row, col, sqaure (features) to softmax probability
feed forward to softmax layer

becomes probabilities after loss
combine 3 outputs then feed forward then loss


num_epochs:
 - 3: 74
 - 5: 76
 - 10: 78

num_filters:
 - 3: 72
 - 6: 71
 - 9: OOM

num_blocks:
 - 10: 72
 - 12: 74
 - 15: 81
 - 18: 78
 - 20: 74

 dropout:
  - 0.2: 66
  - 0.05: 





