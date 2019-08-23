## [**"Homography Estimation from Image Pairs with Hierarchical Convolutional Networks"**](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Nowruzi_Homography_Estimation_From_ICCV_2017_paper.pdf)

Created by [Erlik Nowruzi](http://www.site.uottawa.ca/~fnowr010/), [Robert Lageniere](http://www.site.uottawa.ca/~laganier/), and [Nathalie Japkowicz](https://www.american.edu/cas/faculty/japkowic.cfm).


**Citation:**
```
@InProceedings{Nowruzi_2017_ICCV_Workshops,
author = {Erlik Nowruzi, Farzan and Laganiere, Robert and Japkowicz, Nathalie},
title = {Homography Estimation From Image Pairs With Hierarchical Convolutional Networks},
booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
month = {Oct},
year = {2017}}
```


**To run the code please follow the following steps:**
1. Clone the repository and enter the project root
2. Download the MSCOCO Data set and place it at "../Data/MSCOCO"
3. Process dataset by executing "python3 dataset_prepare.py" to train/test tfrecords
4. Run the training by executing "python3 train_main.py" - will produce log folders at "../Data/logs/"
5. Run the tests by executing "python3 test_main.py"
6. Running the "python3 write_tfrecords.py" will produce the tfrecords of trained model to be used for the next CNN in the chain.
