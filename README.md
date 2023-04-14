<div align="center">
  <a>
    <img src="frontend/assets/signlingo-header.png" alt="Signlingo-Header">
  </a>
  <h3 align="center">Singlingo - Duolingo for SgSL </h3>
  <h3> Made by IS4242 Group 4  </h3>
</div>

## Members 

| Name      | Email |
| ----------- | ----------- |
|Hnin Azali  | e0424743@u.nus.edu |
|Lu Xinyi  | e0421231@u.nus.edu  |
|Melvin Dio Haryadi  | e0550457@u.nus.edu |
|Putri Darmawan  | e0407698@u.nus.edu |
|Wang Ziyue  | e0415688@u.nus.edu |
|Wong Zhou Wai | e0412934@u.nus.edu |

<br>

## Installation

Install the necessary packages by running the following command in the terminal

`pip install -r requirements.txt`

To run the app, use command

`streamlit run frontend/Playground.py`

---
<b> Note </b>
1. Code has been tested on both Windows and Mac with `openCV===4.6.0.66` and `Requirements.txt` was build using a windows laptop.

2. `tensorflow-intel` was used in this build. If you are using a Macbook with <b> M1 or M2 chip </b>, you will need to manually install `tensorflow-macos` instead. 
You may refer to https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706

3. Application's capture function utilises `openCV` capture feature and speed of capture is <b> heavily dependent </b> on your laptop's processing speed. 

<br>

## Folders and files

### <b>data</b>

Contains preprocessed data used to train model

### <b>frontend </b>

Contains files used to run the streamlit application

### <b>Model Development </b>

Contains files used to develop the models used in the streamlit application

#### `cnn.ipynb`

File used to develop CNN model with and without the use of mediapipe

#### `knn_model_mp.ipynb`

File used to develop KNN model with the use of mediapipe

#### `knn_model.ipynb`

File used to develop KNN model without the use of mediapipe

#### `SVM.ipynb`

File used to develop SVM model with the use of mediapipe

#### `SVM_2.ipynb`

File used to develop SVM model without the use of mediapipe



<br>


## Acknowledgements

We would like to thank Assistant Professor Chenshuo Sun, Ms Phoebe Jia Jia Chua of the National University of Singapore for the opportunity to embark on this project. We would also like to thank the volunteers for providing their time and effort into the creation of our dataset which has provide invaluable to our developmental efforts. 

<br><br>


<sub>
Copyright (C) 2023. This project was created using information collected from publicity available videos and volunteers and was created for educational reasons. Any parts of this project should ONLY be used for NON-COMMERICAL reasons. This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses.
</sub>

