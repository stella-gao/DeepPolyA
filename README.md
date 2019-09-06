# DeepPolyA
## Collect the positive data and negative data
There are two data file samples in the data/ folder: positive_sample.fa, negative_sample.fa
(The data is in FASTA format)


## Process the data
In the codes/ folder, process the data with dna_io_1mer.py and dna_io.py 
Then change data into HDF5 format as the input for deep learning with hdf5-train.py, hdf5-valid.py and hdf5-test.py


## Train, test and evaluate the model 
Run deeppolyA.py
There are several hyperparameters which can be turned during training, such as convolution kernel size, number of layers, dropout rate, learning rate, the number of epochs, etc.


## Run the program at the background
```shell
nohup python deeppolyA_v1.py > log.txt 2>&1 &


