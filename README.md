Deep Homography Estimation with TensorFlow using python - Version 1.0 - Clean Architecture

To run the code please follow the following steps:
1. Clone the repository and enter the project root
2. Download the MSCOCO Data set and place it at "../Data/MSCOCO"
3. Process dataset by executing "python3 dataset_prepare.py" to train/test tfrecords
4. Run the training by executing "python3 train_main.py" - will produce log folders at "../Data/logs/"
5. Run the tests by executing "python3 test_main.py"
6. Running the "python3 write_tfrecords.py" will produce the tfrecords of trained model to be used for the next CNN in the chain.
