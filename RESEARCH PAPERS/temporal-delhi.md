www.nature.com/scientificreports 



## **OPEN Temporal trends and predictive modeling of air pollutants in Delhi: a comparative study of artificial intelligence models** 

**Omer A. Alawi**<sup>**1,2**</sup> **, Haslinda Mohamed Kamar**<sup>**1**</sup> **, Ali Alsuwaiyan**<sup>**3,4**</sup> **& Zaher Mundher Yaseen**<sup>**5**</sup> 

**Air pollution monitoring and modeling are the most important focus of climate and environment decision-making organizations. The development of new methods for air quality prediction is one of the best strategies for understanding weather contamination. In this research, different air quality parameters were forecasted, including Carbon Monoxide (CO), Nitrogen Monoxide (NO), Nitrogen Dioxide (NO2), Ozone (O3), Sulphur Dioxide (SO2), Fine Particles Matter (PM2.5), Coarse Particles Matter (PM10), and Ammonia (NH3). Hourly datasets were collected for air quality monitoring stations near Delhi, India, from November 25, 2020 to January 24, 2023. In this context, five intelligent models were developed, including Long Short-Term Memory (LSTM), Bidirectional Long-Short Term Memory (Bi-LSTM), Gated Recurrent Unit (GRU), Multilayer Perceptron (MLP), and Extreme Gradient Boosting (XGBoost). The modelling results revealed that Bi-LSTM model had the best predictability performance for forecasting CO with (R**<sup>**2**</sup> **= 0.979), NO with (R**<sup>**2**</sup> **= 0.961), NO2 with (R**<sup>**2**</sup> **= 0.956), SO2 with (R**<sup>**2**</sup> **= 0.955), PM10 with (R**<sup>**2**</sup> **= 0.9751) and NH3 with (R**<sup>**2**</sup> **= 0.971). Meanwhile, GRU and LSTM models performed better in forecasting O3 and PM2.5 with (R**<sup>**2**</sup> **= 0.9624) and (R**<sup>**2**</sup> **= 0.973), respectively. The current research provides illuminating visuals highlighting the potential of deep learning to comprehend air quality modeling, enabling improved environmental decisions.** 

**Keywords** Air quality forecasting, Air pollution monitoring, Deep learning, Particulate matter, Environmental assessment 

#### **General background of study** 

Climate change has been a significant issue recently due to its impact on weather conditions and land temperatures, which individuals or natural phenomena can cause. The main causes of air pollution include coal and gas combustion for power generation, transportation, industrial and residential developments<sup>1</sup> . Global warming is caused by greenhouse gases (GHG), while climate change is caused by global warming. Emerging nations such as India are facing numerous issues related to air pollution and its detrimental environmental and public health consequences<sup>2</sup> . India is experiencing substantial concerns regarding air quality degradation, such as the massive population growth, industrial companies, the use of fossil fuels for power generation, poor agricultural methods, and motor vehicle emissions<sup>3,4</sup> . Some particulate matter and gaseous pollutants are produced directly from the source and cause air pollution such as in the size of 2.5 and 10 microns (PM2.5 and PM10), carbon dioxide (CO2), Sulphur dioxide (SO2), Nitrogen oxide (NO2), Ammonia (NH3), benzene, volatile organic compounds (VOCs), carbon monoxide (CO), and ozone (O3)<sup>5</sup> . The initial air pollutants are used in the general equation to calculate the air quality index (AQI), depending on the geographical region. Besides, these pollutants are the cause of the following issues such as air pollution, depletion of ozone layer, global warming, increased average land temperature, climate change, and acid rain<sup>6</sup> . The air pollutants concentration changes based on the main atmospheric factors such as precipitation (snowfall, rain, sleet or ice pellets, drizzle, hail, freezing rain, frost, and rime), wind speed (WS), wind direction (WD), relative humidity (RH), solar radiation 

1Department of Thermofluids, Department of Mechanical Engineering, Universiti Teknologi Malaysia, 81310 UTM Skudai, Johor Bahru, Malaysia.<sup>2</sup> Department of Power Mechanics Engineering Techniques,Technical Engineering College, Al- Bayan University, Baghdad 10011, Iraq.<sup>3</sup> Department of Computer Engineering, King Fahd University of Petroleum and Minerals, Dhahran 31261, Saudi Arabia.<sup>4</sup> Interdisciplinary Research Center for Intelligent Secure Systems, KFUPM, Dhahran, Saudi Arabia.<sup>5</sup> Civil and Environmental Engineering Department, King Fahd University of Petroleum & Minerals, Dhahran 31261, Saudi Arabia.<sup></sup> email: z.yaseen@kfupm.edu.sa; zaheryaseen88@gmail.com 

**Scientific Reports** |        (2024) 14:30957 

1 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 

(SR), and air temperature (T)<sup>7</sup> . In this context, AI approaches such as machine learning and deep learning can be developed using various datasets collected from different monitoring sites and stations to create strong connections between inputs and outputs. 

#### **Literature review** 

As per the open literature<sup>8,9</sup> , complex AI models were developed by various researchers to predict the air pollutant levels<sup>10,11</sup> . For instance, the air quality index of four stations such as New Delhi, Bangalore, Kolkata, and Hyderabad, was predicted by using three different regression algorithms such as support vector regression (SVR), random forest (RF), and CatBoost (CR)<sup>12</sup> . The RF model showed the lowest root mean square error (RMSE) in Bangalore, Kolkata, and Hyderabad, with a value of 0.5674, 0.1403, and 0.3826, respectively. Meanwhile, the CR model obtained the lowest RMSE value (0.2792) in New Delhi. Additionally, the CR was the superior model in terms of accuracy in New Delhi (R<sup>2</sup> = 79.8622%) and Bangalore (R<sup>2</sup> = 68.6860%), respectively. Two different approaches such as deep learning and Holt-Winters statistical model were compared to predict the PM2.5 and PM10 concentrations<sup>13</sup> . The Holt-Winters statistical model exhibited lower RMSE and MSE values than the deep learning model. Also, the PM2.5 concentration was forecasted using LSTM, which was integrated to the particle swarm optimization (PSO) method<sup>14</sup> . They collected their datasets from 15 monitoring stations and achieved a strong accuracy ranging from R<sup>2</sup> = 0.86 to R<sup>2</sup> = 0.99. Also, the training and testing sets showed an average percentage error over 15 locations of about 6.6% and 6.9%, respectively. In Chennai, the AQI values were classified using a novel expert system integrated from support vector regression (SVR) and LSTM algorithms<sup>15</sup> . Their novel expert system showed a superior performance with R<sup>2</sup> = 0.97 and RMSE = 10.9. 

A case study in Delhi, a dataset was collected from January 2018 to October 2021 to predict the PM2.5 concentration using deep learning model that combined neural networks, fuzzy inference systems (ANFIS), and wavelet transforms<sup>16</sup> . Their novel model showed an outstanding accuracy, such as: 0.95 < R<sup>2</sup> < 0.99 for 1-day short term prediction, 0.85 < R<sup>2</sup> < 0.94 for 2-day short term prediction, and 0.81 < R<sup>2</sup> < 0.93 for 3-day short term prediction. A novel hybrid model from GRU and LSTM was developed to predict PM2.5 concentration in Delhi<sup>17</sup> . Also, the dataset was validated using five different standalone models, such as LSTM, linear regression (LR), GRU, K-Nearest Neighbour (KNN), and support vector machine (SVM). The LSTM-GRU was superior to the individual models with an MAE-value of 36.11 and R<sup>2</sup> -value of 0.84. Moreover, the AQI values were predicted in Chennai using historical data from meteorological locations from 2017 to 2022<sup>6</sup> . The authors developed four different tree-based models such as XGBoost, RF, Bagging Regressor, and LGBM. XGBoost model showed the following prediction metrics: R<sup>2</sup> = 0.9935, MAE = 0.02, MSE = 0.001, and RMSE = 0.04. Twelve pollutants and ten meteorological parameters from July 2017 to September 2022 were collected over Visakhapatnam, Andhra Pradesh, India<sup>18</sup> . This dataset was used to estimate the AQI value using five models such as LightGBM, RF, CatBoost, Adaboost, and XGBoost. The CatBoost model showed the following metrics: R<sup>2</sup> = 0.9998, MAE = 0.60, MSE = 0.58, and RMSE = 0.76. Meanwhile, the Adaboost model presented the following metrics with an R<sup>2</sup> = 0.9753. Encoder-Decoder (ED) layers were connected to GRU deep learning model for predicting 1-hour, 8-hour, and 24-hour of PM2.5 concentrations in New Delhi, India, and the dataset was collected from 2008 to 2010<sup>19</sup> . The hybrid method showed superior performance over the standalone models such as (RF, XGBoost, ANNs, and LSTM). Standalone versus stacking models were used to estimate 1-hr and 24-hr PM2.520. XGBoost showed higher accuracy than RF and LightGBM with R<sup>2</sup> = 0.73. Meanwhile, stacked model (XGBoost as a metaregressor) improved the accuracy of standalone XGBoost with R<sup>2</sup> = 0.77. The eastern region exhibited the best 1-hr prediction with R<sup>2</sup> = 0.80 and substantial reduction in Mean Bias (MB = − 0.03 µg m<sup>− 3</sup> ), followed by the northern region with R<sup>2</sup> = 0.63 and MB = − 0.10 µg m<sup>− 3</sup> . In Chandigarh, eight AI models such as RF, KNN, LR, LASSO regression, Decision Tree (DT), SVR, XGBoost, and Deep Neural Network (DNN) with 5-layers were used to predict 24-hr air pollution and outpatient visits for Acute Respiratory Infections (ARI)<sup>21</sup> . On ARI patients, the RF model performed best, with R<sup>2</sup> = 0.606, 0.608 without lag, and 24-hr lag, respectively. Also, on total patients, R<sup>2</sup> = 0.872, 0.871 without lag, and 24-hr lag, respectively. 

China and India, covering 35% of the global population, face widespread urban air pollution, and real-time and remote sensing assessments remain insufficient. Air quality issues in these rapidly growing economies are increasingly being addressed by the application of machine and deep learning in recent studies. Six models i.e., MLR, SVR, RF, ANN, XGBoost, and LSTM, were used to predict LST for Hyderabad city, India using fiveyear (2018–2022) data on air pollution and meteorological parameters (from ambient air quality monitoring stations) and MODIS LST data<sup>22</sup> . Considerable influence of PM2.5 and CO (during summer) and SO2 (during winter) on LST was observed which demonstrated high sensitivity of these parameters on LST. ANN method demonstrated better accuracy with lower error metrics, comprising of RMSE, MAPE, and MSE, compared to the other approaches with ranking in the order ANN > RF > SVR > XGBoost > LSTM > MLR. Based on hourly observations from  2018 in India, integrating temporal and regional features into the LightGBM model led to a notable enhancement in its performance, achieving a 21% reduction in RMSE for PM2.5 estimation and a 19% reduction for PM1023. The ML model predicted an annual nationwide concentration of 68.3 µg/m3 for PM2.5, which was consistent with high satellite aerosol optical depth (AOD) values. A real-time assessment of hazardous atmospheric pollutants across cities in China (Shanghai, Nanjing, Jinan, Zhengzhou and Beijing) and India (Kolkata, Asansol, Patna, Kanpur and Delhi) was conducted using ground observations, Sentinel-5P and NASA satellite data from 2012 to 2023<sup>24</sup> . GMAO’s SO2, NO2 and CO predictions showed high accuracy with near-perfect PC values and low NRMSE proving model reliability. A Unified Spectro-Spatial Graph Neural Network (USS-GNN) designed for forecasting O3-NO2 concentrations for New Delhi, utilized hourly observations for the years 2021 and 2022<sup>25</sup> . The proposed model achieved R<sup>2</sup> values of 0.650 and 0.618, RMSE of 13.950 and 16.120 µg/m<sup>3</sup> , MAE of 10.730 and 12.930 µg/m<sup>3</sup> for O3 and NO2 , respectively. Different artificial intelligence models were proposed to simulate climate parameters (1 January 1951–31 December 2022) of Jinan city in China, include ANN, RNN, LSTM, CNN, and CNN-LSTM<sup>26</sup> . The hybrid CNN-LSTM model significantly 

**Scientific Reports** |        (2024) 14:30957 

2 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 

reduced the forecasting error compared to the models for the one-month time step ahead. The RMSE values of the ANN, RNN, LSTM, CNN, and CNN-LSTM models for monthly average atmospheric temperature in the forecasting stage were 2.0669, 1.4416, 1.3482, 0.8015 and 0.6292 °C, respectively. 

#### **Research objectives and novelty** 

This study addresses the critical need for an innovative approach to air quality forecasting, focusing on predicting concentrations of eight key air pollutants that significantly impact human health, environmental integrity, and everyday life. The primary objectives are twofold: (1) to develop accurate forecasting models for Carbon Monoxide (CO), Nitrogen Monoxide (NO), Nitrogen Dioxide (NO₂), Ozone (O₃), Sulphur Dioxide (SO₂), Fine Particulate Matter (PM₂.₅), Coarse Particulate Matter (PM₁₀), and Ammonia (NH₃); and (2) to rigorously assess and compare the predictive performance of five advanced standalone artificial intelligence (AI) models tailored for univariate pollutant forecasting. 

The novelty of this research lies in its comparative analysis across five AI models—Long Short-Term Memory (LSTM), Bidirectional Long-Short Term Memory (Bi-LSTM), Gated Recurrent Unit (GRU), Multilayer Perceptron (MLP), and Extreme Gradient Boosting (XGBoost)—in forecasting hourly pollutant concentrations in a high-pollution urban setting. Using extensive data from monitoring stations in Delhi, collected between 25/11/2020 and 24/01/2023, the study performs in-depth statistical analysis to explore patterns and dynamics of each pollutant. Through the use of scatter plots, Taylor diagrams, and forecasting performance matrices, this study provides comprehensive insights into the distinct strengths and limitations of each model. 

In highlighting model efficiency and comparing forecast accuracy metrics, this study provides valuable guidance for advancing air quality management strategies. The findings provide practical implications for urban planning and environmental policy, demonstrating that accurate AI-driven forecasting models can be instrumental in environmental mitigation efforts and health risk assessments in rapidly urban areas. 

### **Data collection and description** 

As outlined above, the datasets of eight air pollutants were collected from the capital city of Delhi, India, spanning from 25/11/2020 to 24/01/2023 including the following elements: Carbon Monoxide (CO), Nitrogen Monoxide (NO), Nitrogen Dioxide (NO2), Ozone (O3), Sulphur Dioxide (SO2), Fine Particles Matter (PM2.5), Coarse Particles Matter (PM10), and Ammonia (NH3). The dataset contains features recorded at an hourly interval, with 18,776 samples. It appears that the datasets are complete, and no missing values were observed. Table 1 shows summary statistics for the eight air pollutants, using various measurements such as Mean, Standard Deviation, Variance, Skewness, Kurtosis, Coefficient of Variation, Median, Interquartile Range (Q3 - Q1), Range (Maximum - Minimum), Median Absolute Deviation, and Robust Coefficient of Variation. Besides, Fig. 1 shows the numerical distributions of eight air pollutants through boxplot formats. Boxplots highlighted extreme outliers that significantly changed the common relationship of the dataset. In this regard, the Winsorizing technique was implemented to manage outliers by replacing extreme values with values closer to the upper or lower limits, resulting in a more robust and reliable analysis of the air quality data. 

### **Applied artificial intelligence models** 

This section outlines the five models employed for predicting air pollutant concentrations. The development and configuration processes of the models are then reviewed. Finally, the section concludes with a summary of the forecasting metrics used to evaluate the accuracy of the model predictions. 

#### **Multilayer perceptron (MLP)** 

An MLP is a fundamental component of ANNs<sup>27</sup> . It can model complex systems in engineering and data sciences. An MLP is characterized by the number of layers of interconnected neurons and the number of neurons per layer. Every layer transforms its input data and feeds it to the next layer in a purely feedforward fashion, contributing to the network’s decision-making process<sup>28</sup> . Figure 2a illustrates an MLP with an input layer, a single hidden layer, and an output layer. Every neuron implements a weighted sum with a bias term fed to an activation function, e.g., tanh. Figure 2b shows the structure of a neuron<sup>29</sup> . 

The architecture of an MLP, including the number of layers and neurons in every layer, significantly affects its performance. Additionally, the use of hyperparameters, such as the learning algorithm, is crucial in training MLPs to minimize their prediction errors effectively. The objective of the learning algorithm is to solve for the weights and biases that minimize an objective function<sup>30</sup> . MLPs are capable of capturing non-linear data, making them suitable for various applications, including air pollutant prediction. One crucial property of MLPs is that the signals propagate from the inputs to the outputs in one direction, i.e., MLPs do not contain feedback loops. This property of MLPs limits them from capturing long-lasting relationships of time series. This is precisely why the prediction error rates are higher than the other models. 

#### **Long short-term memory (LSTM)** 

To understand the improvements offered by Long Short-Term Memory (LSTM) networks, the limitations of recurrent neural networks (RNNs) are first examined, as these are addressed by the LSTM architecture. RNNs contain both feedforward and feedback loops that facilitate capturing complicated temporal dynamics of sequential time series data<sup>27</sup> . The main limitation of RNNs is that they fail to capture long-term dependency in the time series, and this is due to the vanishing and exploding gradients<sup>27</sup> . 

Schmidhuber and Hochreiter<sup>31</sup> proposed the LSTM model in 1997 to solve the issue of vanishing and exploding gradients in RNNs. One of the proposed ideas is the cell state, which is intended to enhance the hidden state and to propagate the effect of some data through longer sequences. Also, LSTM introduces three 

**Scientific Reports** |        (2024) 14:30957 

3 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 



<!-- Start of picture text -->
Robust coefficient of variation 0.784 1.483 0.638 1.483 0.601 0.872 0.810 0.785<br>Median  absolute  deviation 974.660 5.250 23.300 27.180 21.460 92.630 114.565 9.250<br>Range  (maximum– minimum) 20888.330 500.680 456.340 801.090 574.580 1696.260 1954.860 287.770<br>Interquartile range  (Q)–Q31 2616.880 35.080 49.700 92.640 47.210 228.560 269.205 20.770<br>Median 1842.500 5.250 54.150 27.180 52.930 157.445 209.705 17.480<br>Coefficient of variation 0.975 1.846 0.733 1.333 0.741 0.951 0.890 1.051<br>Kurtosis 4.340 9.087 6.487 5.744 12.074 4.456 4.023 18.752<br>Skewness 2.005 2.808 2.034 1.997 2.670 1.949 1.868 3.568<br>Variance 8148304.444 3859.779 2354.917 6474.605 2444.234 51317.483 71377.579 697.071<br>Standard  deviation 2854.524 62.127 48.527 80.465 49.439 226.534 267.166 26.402<br>Mean 2929.229 33.661 66.221 60.346 66.694 238.130 300.093 25.110<br> missing N 0 0 0 0 0 0 0 0<br> total N 18,776 18,776 18,776 18,776 18,776 18,776 18,776 18,776<br>CO NO NO2 O3 SO2 PM2.5 PM10 NH3<br><!-- End of picture text -->

**Scientific Reports** |        (2024) 14:30957 

4 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 



<!-- Start of picture text -->
2000 600<br>25% 75% 25% ~75%<br>50<br>— Median Line = Median Line<br>20000 I Range within 1.51QR| J Range within 1.51QR|<br>* Outliers + Outliers<br>15000 © Mean ‘400 O Mean<br>= "E 300<br>° rsan02 S 200<br>i} 5000 4<br>saa 100 02<br>0 260.35SS oos12 ° 35.76ps<br>-5000 -100<br>500 75%-75%<br>R 25%-75% 800 25%~15%<br>Range within 1.S1QR| ' Range within 1.51QR|<br>400 = Median Line «| Median Line<br>Mean 3 | Mean<br>= 0 . Outliers _ 600 + Outliers<br>& E<br>=200 B 400<br>S 157.68 a<br>4 100 © 21.74<br>863 200<br>0 2 92.98<br>0 os<br>-100<br>700 2000<br>600 }25%~75% 25% ~75%<br>— MedianRange within Line 1.SIQR) 1600 =LT RangeMedianwithinLine 1.51QR)<br>500 O Mean O Mean<br>Z § + Outliers ~ 1200 + Outliers<br>5 400 “a<br>EW= 300 3== 800<br>F 200 =<br>a 152.59 e400<br>100 s202<br>0 525 saat 0<br>-100 -400<br>2500 350<br>25% 75% 25%~75%<br>2000 | RangeMedianwithin Line 1.51QR| 300 3] —T RangeMedianwithinLine 1.51QR|<br>O Mean 250 O Mean<br>+ Outliers * Outliers<br>~ 1500<br>* = 200<br>2 &<br>= 1000 = 150<br>=== 50 791.17 %=z: 100<br>38799<br>.<br>0 1507 118,785 so ee 04<br>0 ° a<br>-500 “50<br><!-- End of picture text -->

**Fig. 1** . Boxplots distributions of eight air pollutants. 

elements that are intended to filter the data flowing through the network in such a way to prevent the vanishing and exploding gradients, and thus maintain that series of long-term dependence. These elements are known as input gate, forget gate, and output gate, as shown in Fig. 3. In particular, the forget gate selects what information to remember or forget. The value of the forget gate is a value between 0 and 1, where 0 means to forget the information and 1 means to remember, and a value in between means partially forget/remember. 

**Scientific Reports** |        (2024) 14:30957 | https://doi.org/10.1038/s41598-024-82117-z 

5 

www.nature.com/scientificreports/ 



<!-- Start of picture text -->
“Tnput | ‘o. oe<br>‘| Layer | A | ‘Output<br>| GRAZSOs SfXAD 7<br>meteOn. ©) im<br>| | GRASSRRS JPRVTS pL<br>posses C) | 1 LQwe<br>(a) (b)<br><!-- End of picture text -->

**Fig. 2** . ( **a** ) An example architecture of an MLP consisting of an input layer, a single hidden layer, and an output layer of neurons ( **b** ) An example architecture of a neuron with two inputs and tanh activation function. 



<!-- Start of picture text -->
| a i<br>| j i 4 \<br>Os on an ae<br>| i<br>1 gf 1 i}<br>' IC) E4©P t GO |<br>mn| ©) HOMMO}Life LS$}!ie IOf_-\ eeLE!si<br>--- 2 |<br><!-- End of picture text -->

**Fig. 3** . Detailed LSTM Cell Architecture. Arrows indicate the flow of information, including the feedback loops for the cell state ( _Ct_ ) and hidden state ( _ht_ ). Note that the summation nodes include bias terms. 

#### **Bidirectional long-short term memory (Bi-LSTM)** 

As proposed by Schuster and Paliwal in<sup>32</sup> , Bi-LSTM operates two instances of LSTM, one for processing the sequence in the forward direction and the other for processing the sequence in the reverse direction, enhancing the model to learn from the sequence in both directions. Bi-LSTM then combines the responses of both directions to generate a unified output sequence that can be used to predict data more accurately<sup>33</sup> . 

**Scientific Reports** |        (2024) 14:30957 

6 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 

#### **Gated recurrent unit (GRU)** 

GRU improves LSTM by reducing the number of gates from three to two<sup>34</sup> . One gate is the update gate, and the other is the reset gate. The primary objective of these gates is to capture the long-term dependency between the data elements of a time series. The main objective of the update gate is to determine how many dependencies to remember, whereas for the reset gate, the main objective is to determine how many dependencies to forget. GRU utilizes a sequence of computations to generate the new hidden state, given the input and the current hidden state. This sequence of computations, i.e., Eqs. (1)–(4), shown in the following table, is then repeated in the form of a loop, to improve the accuracy of GRU to predict new data and to effectively capture the long-lasting relationship between the elements of the time series. 

The update gate decides how much to remember 



The reset gate decides how much to forget 



The candidate hidden state computation according to reset gate feedback 



The new hidden state computation, according to feedback from the update gate and candidate hidden state 



#### **Extreme gradient boosting (XGBoost)** 

All the previous models are proposed to optimize the performance of the gradient boosting machine (GBM)<sup>35</sup> , XGBoost integrates a number of weak learners to form an efficient predictive model as illustrated in Fig. 4<sup>36</sup> . XGBoost generates _k_ decision trees and validates them using a subset of the training data such that the model generalizes well to new data. Each decision tree creates a residual, i.e., the error in prediction of the current 

# a 

**Fig. 4** . XGBoost architecture consists of k decision trees that iteratively improves the prediction accuracy. 

**Scientific Reports** |        (2024) 14:30957 

7 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 

decision tree, and this residual becomes the target variable for prediction in the next decision tree. XGBoost has been proven to be effective in predictions requiring high levels of accuracy, e.g., see<sup>28,37–43</sup> . 

#### **Models development and configuration** 

Figure 5 shows the development and configuration process. The implementation platform was Google Collaboratory, utilizing Python and powered by GCC 9.4.0. Many libraries were utilized to complete this project, including NumPy, Pandas, Scikit-learn, and the APIs for TensorFlow Keras. The input dataset contains the hours of concentrations of eight air pollutants in Delhi between November 25th, 2020, and January 24th, 2023. Prior to any actions being executed on the input dataset, it was standardized using scikit-learn and then split into 80–20% training and testing sets. 

The training set uses a filter to substitute outliers by values that are closer to the upper or lower bound of the dataset when these outliers are excluded. After that, the filtered data is processed to each of the five models to complete the learning process. The test data is utilized to evaluate the accuracy of the trained models by generating a set of six performance metrics, such as RMSE, R<sup>2</sup> , MAE, MSE, standard deviation and WI . 

Prior to the training, hyperparameter optimization and tuning were conducted to maximize the performance of each model. This tuning process was conducted using scikit-learn algorithms specifically developed for such purposes. The GridSearchCV algorithm was employed for neural network-based models, while for the XGBoost model, the RandomizedSearchCV algorithm was employed. The optimized hyperparameters for all models are presented in Table 2. 

#### **Forecasting metrics** 

There are several statistical performance metrics were adopted for the forecasting accuracy evaluation for each developed predictive model. The mathematical expression of each metric are as follows: Coefficient of Determination (R<sup>2</sup> ) 



<!-- Start of picture text -->
! (nput Dataset) I<br>! | python version 3.10.12, GCC !<br>1 9.4.0, Google Collaboratorative 1<br>1 | notebook, many libraries, I<br>|_| including NumPy, Pandas, | | A datasetof the hourly levels of the shown airborne particles from monitoring stations around Delhi|<br>I scikit-learn, and TensorFlow's for the period from November 25, 2020 to January 25, 2023, amounting to 18,776 data samples. i<br>1Keras API.<br>1= 1<br>'Standadize data usini I<br>f ‘StandardScalar of scikit-learn 1<br>!<br>ry<br>' a II<br>1 Training Dataset “Split dataset into 80% tr 7 Test Dataset !<br>| ‘set and 20% testing set. !<br>} Forecasting Evaluation) ll 11<br>}!<br>!<br>11<br>! (Winsorizer Filter) !<br>| Gifinsorizing involves replacing extreme '<br>| [values with values closer to the upper or Filtered !<br>I1 \_ andlower reliablelimits, analysiscontributing of theto aira quality data.more robust Data i1<br>i<br>!! (iyperparameters Tuning) |<br>| PerformedRandomizedSearchCVusing GridSearchof scikit-or XGBoost '!<br>1 learn. 1<br>!<br>11<br><!-- End of picture text -->

**Fig. 5** . Models development and configuration. 

**Scientific Reports** |        (2024) 14:30957 

8 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 



<!-- Start of picture text -->
Model Hyperparameter values tested Best value<br>n_estimators [50, 100, 150] Best: 100<br>learning_rate [0.01, 0.1, 0.2] Best: 0.1<br>XGBoost<br>max_depth [3, 5, 7] Best: 5<br>min_child_weight [1, 3, 5] Best: 1<br>neurons [50, 100] Best: 50<br>LSTM epochs [50, 100] Best: 50<br>batch_size [32, 64] Best: 32<br>neurons [50, 100] Best: 50<br>Bi-LSTM epochs [50, 100] Best: 50<br>batch_size [32, 64] Best: 32<br>neurons [50, 100] Best: 50<br>GRU epochs [50, 100] Best: 50<br>batch_size [32, 64] Best: 32<br>neurons [50, 100] Best: 50<br>MLP epochs [50, 100] Best: 50<br>batch_size [32, 64] Best: 32<br><!-- End of picture text -->

**Table 2** . Hyperparameters, along with the best values, were tested for LSTM, Bi-LSTM, GRU, ANN-MLP, and XGBoost models. 



Root Mean Square Error (RMSE) 

Mean Absolute Error (MAE) 



Mean Squared Error (MSE) 



Standard Deviation (σ) 



Willmott Index (WI) 



where _y_ � is the predicted air pollutant, _y_ is the actual air pollutant, _mean_ ( _y_ ) is the mean of actual target output, σ is the Standard Deviation, _µ_ is the air pollutant mean, _N_ is the number of observations. 

### **Results and discussion** 

This section presents the results and analysis of various ML models (LSTM, Bi-LSTM, GRU, MLP, and XGBoost) applied to forecast concentrations of eight airborne particles: Carbon Monoxide (CO), Nitrogen Monoxide (NO), Nitrogen Dioxide (NO₂), Ozone (O₃), Sulphur Dioxide (SO₂), Fine Particulate Matter (PM₂.₅), Coarse Particulate Matter (PM₁₀), and Ammonia (NH₃). Each model was evaluated both graphically and statistically. To assess the predictive performance, a Taylor diagram was used to compare the models based on three key statistical metrics—correlation, RMSE, and standard deviation—in relation to the benchmarked observational 

**Scientific Reports** |        (2024) 14:30957 

9 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 

dataset. Additionally, scatter plots were generated to visualize the deviations between actual and forecasted values along the 1:1 line. 

#### **Carbon monoxide (CO)** 

Carbon monoxide was forecasted due to its highly poisonous concern for human health<sup>44</sup> . Figure 6 presents a scatter plot and Taylor diagram along with the forecasting metrics. According to the scatter plot, the CO concentration was varying between (0-23000). In general, all models showed a good variation around the 45° line. However, MLP and XGBoost revealed some observations from the identical line. According to Fig. 6a, the obtained determination coefficient based on the regression scatter formula was (LSTM = 0.97, Bi-LSTM = 0.97, GRU = 0.97, MLP = 0.94 and XGBoost = 0.96). Moreover, the Taylor diagram revealed that the observational dataset of the CO was located at the standard deviation of 1 as per Fig. 6b. Overall, Bi-LSTM surpassed others with the highest R<sup>2</sup> (0.979), lowest RMSE (438.898), and lowest MAE (244.543), indicating precise and reliable predictions (Fig. 6c). GRU also performed well, with an R² of 0.977 and RMSE of 456.729. LSTM followed closely with an R² of 0.971. XGBoost and MLP had a relatively lower accuracy, with MLP having the lowest R<sup>2</sup> (0.941) and the highest RMSE (740.927). Therefore, Bi-LSTM demonstrated the best performance, making it the most effective model for accurate air quality forecasting in this research. 

#### **Nitrogen monoxide (NO)** 

NO is an extremely reactive gas that is initiated during high-temperature fuel burning. It is emitted by automobiles and non-road vehicles (e.g., boats and construction equipment). Breathing NO with a high concentration can cause respiratory diseases such as asthma, which could lead to respiratory infections<sup>44</sup> . According to the scatter plots visualization (Fig. 7a), the proposed models differed from the identical best-fit line. However, LSTM, GRU, and MLP demonstrated some observation scatters from a similar line. Numerically, the attained determination coefficient based on the regression scatter formula was (LSTM = 0.94, Bi-LSTM = 0.96, GRU = 0.95, MLP = 0.91 and XGBoost = 0.94). In addition, based on the Taylor diagram presentation (Fig. 7b), the observational dataset of the NO was located at the standard deviation of 1. In this comparison of air quality forecasting models (Fig. 7c), Bi-LSTM emerged as the top performer with the highest R² (0.961), lowest RMSE (13.922), and MAE (7.854), indicating accurate and consistent predictions. GRU and LSTM followed, with R² values of 0.956 and 



<!-- Start of picture text -->
28000, pone » re)<br>(a) || 0201 a2 9, , ieee7M<br>20000 - 4 z i 135 x os % + MpGRU<br>Ht ii 120}Bs < *, XGBoost<br>| i Yi i 1.05 x Yep,<br>= 15000 00 oieee)eo || 3BPS090 KTP wno %<br>& sc000 Jer.Bor” LSP :i zg — py Se,s 2,<br>od See * | © 0.60 ‘<br>ara 376, Ls & \ A<br>5000 Peet st i OBiLs™ | 0.45 ++ \ \<br>| Re eee cay 030 \ 7<br>ay = OXGBoost | 0.15 &\ Y be 7 8<br>oa i Li r<br>i) 5000 10000 15000 20000 25000 0.0% "9e% ses 8 8 ry Lad<br>CO (Actual) 66 6S 6S 6 oO 6G A A a<br>Standard deviation<br>(c) R RMSE MAE MSE o Wil<br>LSTM 0.971 521.507 289.280 271969.200 519.953 0.975<br>Bi-LSTM 0.979 438.898 244.543 192631.300 437.793 0.983<br>GRU 0.977 456.729 | 260.968 208601.200 452.646 0.981<br>MLP 0.941 740.927 | 463.072 | 548973.400 738.752 (0.951<br>XGBoost 0.963 582.624 315.390 | 339450.600 582.385 0.969<br><!-- End of picture text -->

**Fig. 6** . Actual and forecasted values of CO using different AI-models; ( **a** ) Scatter plots, ( **b** ) Taylor diagram, ( **c** ) Forecasting metrics. 

**Scientific Reports** |        (2024) 14:30957 

10 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 



<!-- Start of picture text -->
600, Po canal<br>(a) | 7M<br>8<br>i 0001 02 9, , ee<br>$00 >, H 1.35 Lo. 6 GRUMup<br>aw H 5 Ss  %, XGBoost<br>ys : 120 Le,<br>o > fei i |<br>~<br>a 2 ORY tas So i £05) + x4 ep<br>4 oChee i g 0 pessse Se,<br>ppt 0, =O i § 090; | 7 rs Ke<br>5 00 o OSs : g a 25<br>8 On 2°OF DrOLEDCASS: :i Sot wasaa °<br>200 £ ee ets : 2 060 » y<br>>, Gareoes = LST™M H & j, \ a. A<br>100 Sto{page“ox e O GRUMLP ii 0.30: a i<br>waste OXGBoost | 0.15} & Sa 4 2<br>~<a H 4 i) 1<br>oe i 6 \ 4 7<br>0 100 = 200» 300-400 $0060 0.08 “™ eeernresger ?<br>NO (Actual) eseee @ So aa nt<br>Standard deviation<br>an) R? RMSE MAE | MSE | o WI<br>LSTM 0.948 16.044 8.273 257.403 15.964 0.960<br>Bi-LSTM 0.961 13.922 7.854 193.832 13.895 0.969<br>GRU 0.956 14.751 8.441 217.582 14.749 0.963<br>__ XGBoostMLP 0.9130.943 20.85216.927 115117.948 434.788286.534 | 20.85016.924 0.9340.954<br><!-- End of picture text -->

**Fig. 7** . Actual and forecasted values of NO using different AI-models; ( **a** ) Scatter plots, ( **b** ) Taylor diagram, ( **c** ) Forecasting metrics. 

0.948, respectively, and slightly higher RMSEs and MAEs. XGBoost performed well, but had a slightly lower accuracy (R<sup>2</sup> = 0.943). The lowest performance was achieved by MLP, with an R<sup>2</sup> of 0.913 and the highest RMSE (20.852). Overall, Bi-LSTM proved the most reliable model for precise air quality forecasting in this study. 

#### **Nitrogen dioxide (NO2)** 

Increases in mortality and hospital admissions for respiratory diseases are also associated with air nitrogen dioxide concentrations<sup>44</sup> . The lungs’ ability to combat microorganisms can be compromised by nitrogen dioxide, leaving the air more vulnerable to illnesses. Additionally, it may cause asthma to be worse. However, LSTM, GRU, and MLP revealed some observation scatters from the identical line. According to (Fig. 8a), the attained determination coefficient based on the regression scatter formula was (LSTM = 0.95, Bi-LSTM = 0.95, GRU = 0.94, MLP = 0.94, and XGBoost = 0.94). The Taylor diagram in Fig. 8b indicated that the observational dataset of the NO2 was located at the standard deviation of 0.90 to 1. In this analysis of air quality forecasting models (Fig. 8c), Bi-LSTM achieved the highest R<sup>2</sup> (0.956) and the lowest RMSE (11.055) and MSE (122.215), suggesting its superior accuracy. Also, LSTM performed well, with an R² of 0.953 and a similar RMSE (11.452), making it a strong contender. GRU followed closely, though with slightly lower accuracy (R² = 0.946) and higher error metrics. XGBoost and MLP had a lower accuracy, with MLP having the lowest R<sup>2</sup> (0.937) and the highest RMSE (13.275). Overall, Bi-LSTM provided the most precise and consistent air quality predictions in this comparison. 

#### **Ozone (O3)** 

Many health problems, such as congestion, coughing, throat irritation, and chest pain, can be caused by breathing in ground-level ozone<sup>44</sup> . Bronchitis, asthma, and emphysema can all become worse due to it. Additionally, ozone can irritate the lining of the lungs and impair lung function. Long-term lung tissue scarring could result from repeated exposure. Increased vulnerability to diseases, pests, and other stressors such as severe weather results from elevated ozone levels, which also reduce the yields of commercial forests and crops and the growth and survival of tree seedlings. As per Fig. 9a, XGBoost and MLP revealed some observation scatters from the identical line. Numerically, the attained determination coefficient based on the regression scatter formula was (LSTM = 0.95, Bi-LSTM = 0.96, GRU = 0.96, MLP = 0.95, and XGBoost = 0.96). The observational dataset of the 

**Scientific Reports** |        (2024) 14:30957 

11 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 



<!-- Start of picture text -->
$00, om canal<br>(a) 0.001 92 os (b) oa<br>400 fy : 1.35 $s a & « ie<br>Yr : 4) xs 2,%, XGBoost<br>09 age : 120) Ne<br>> 300 of OED ® / 1.05 | 4 “5<br>I c i.e i & mi 0.<br>z ODI)mel ops) ii %@> 0.90} y en at Xe<br>SZ, 200 2Ceo.CREPo |i 3z os \ZNSS ; a<br>coteueen oO LSTM H & \ . A<br>1001 EEROCACHES OBiLs™GRU |: 0.309.4577 Xx\ o<br>geALA So OXGBoostMLP i| 0.15 & Sa Lt y °8<br>Pe a ! es \ 4 7<br>0 100 200 300 400 500 0.08 “™ eeernresger ?<br>NO, (Actual) eseee @ So aa nt<br>Standard deviation<br>© | RFR RMSE MAE | MSE | o WI<br>LSTM 0.953 11.452 6.620 131.146 11.434 0.961<br>Bi-LSTM 0.956 11.055 6.675 122.215 11.043 0.964<br>GRU 0.946 12.222 7.028 149.385 12.121 0.959<br>___MLP 0.937 13.275 7.799 | 176.232 13.239 0.949<br>XGBoost 0.940 12.879 7.067 165.871 12.877 0.951<br><!-- End of picture text -->

**Fig. 8** . Actual and forecasted values of NO2 using different AI-models; ( **a** ) Scatter plots, ( **b** ) Taylor diagram, ( **c** ) Forecasting metrics. 

O3 was located at the standard deviation of 1 according to Taylor diagram in Fig. 9b. In this air quality forecasting comparison (Fig. 9c), both GRU and XGBoost excelled with the highest R² values (0.962). GRU achieved a lower RMSE (12.050) and MSE (145.191), while XGBoost had the lowest MAE (6.409), indicating excellent accuracy and minimal error. Bi-LSTM and LSTM also performed well with R² values of 0.959 and 0.958, respectively, but with slightly higher RMSE and MAE values. MLP, while accurate (R² = 0.957), showed the highest RMSE (12.869). In this part, GRU and XGBoost proved the most effective models for accurate and consistent air quality predictions. 

#### **Sulphur dioxide (SO2)** 

The combustion of sulphur-containing fuels releases sulphur dioxide (SO₂), impacting ecosystems and human health. SO₂ harms plants, streams, and forests, and in humans, it worsens respiratory conditions like asthma and bronchitis, especially during exercise<sup>44</sup> . Studies also link SO₂ exposure to higher cardiovascular disease risks. However, XGBoost and MLP revealed some observation scatters from the identical line. As per Fig. 10a, the attained determination coefficient based on the regression scatter formula was (LSTM = 0.95, Bi-LSTM = 0.95, GRU = 0.95, MLP = 0.92, and XGBoost = 0.91). Based on the Taylor diagram in Fig. 10b, the observational dataset of the SO2 was located at the standard deviation of 1 except MLP (SD = 0.86). In this comparison of air quality forecasting models (Fig. 10c), Bi-LSTM achieved the highest performance with the highest R<sup>2</sup> (0.955), the lowest RMSE (11.110), and MAE (6.076), indicating superior accuracy and low error. LSTM also performed well with an R² of 0.950 and slightly higher RMSE (11.648). GRU followed closely, with similar metrics (R² = 0.949). The accuracy of MLP and XGBoost was lower, with MLP achieving an R<sup>2</sup> of 0.928 and XGBoost the lowest R<sup>2</sup> (0.916) and highest RMSE (15.214). In conclusion, Bi-LSTM was the most effective model for precise air quality predictions in this study. 

#### **Fine particles matter (PM2.5)** 

Many scientific investigations have demonstrated that particulate matter decreases visibility and negatively impacts materials, ecosystems, and climate. PM, particularly PM2.5, alters how light is absorbed and scattered in the atmosphere, which can affect visibility<sup>45,46</sup> . Long-term exposure to fine particles may also increase the risk of heart disease and be linked to a higher incidence of chronic bronchitis, deteriorated lung function, and 

**Scientific Reports** |        (2024) 14:30957 

12 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 



<!-- Start of picture text -->
600 nnn = cba<br>(a) oo 01 o2 9, (b) 7 le<br>500 QAo i 1.35 QX05 %, + mi<br>/ wok 42 26%, XGBoost<br>5 e i § “Ta 45 %,<br>3 ° ogg i cy) ee fn<br>[o) wer cade ° i z m °<br>200 OEenna © ii $2 0.60 | ‘ ., \°<br>eee veuu LSTM H é \ 2<br>ue” OBiLsT™ | OAs \ 7<br>100 |» eee GRU i 0.30 \ °<br>+62 OXGBoost | 0.15 by) & % & 8<br>A otie= H es \ 4 i<br>0 100 200» 300, 400s S00 600 00a 2 he ne Mo he<br>O; (Actual) géegaReseraggagee 59 5S 5 oS oO a a a<br>Standard deviation<br>(c) R RMSE MAE MSE o WI<br>LSTM 0.958 12.754 7.820 162.659 12.400 0.967<br>Bi-LSTM 0.959 12.612 7.046 159.054 | 12.573 0.967<br>GRU 0.962 12.050 6.729 145.191 11.988 0.972<br>MLP 0.957 12.869 7.867 165.608 12.865 0.966<br>XGBoost 0.962 12.136 6.409 147.279 12.125 0.969<br><!-- End of picture text -->

**Fig. 9** . Actual and forecasted values of O3 using different AI-models; ( **a** ) Scatter plots, ( **b** ) Taylor diagram, ( **c** ) Forecasting metrics. 

lung cancer, according to studies. However, XGBoost and Bi-LSTM revealed some observation scatters from the identical line. According to Fig. 11a, the attained determination coefficient based on the regression scatter formula was (LSTM = 0.97, Bi-LSTM = 0.97, GRU = 0.96, MLP = 0.93, and XGBoost = 0.96). The Taylor diagram illustration (Fig. 11b) demonstrated that, the observational dataset of PM2.5 was located at the standard deviation of 1.05. In this analysis of air quality forecasting models (Fig. 11c), LSTM demonstrated the best performance with the highest R² (0.973), the lowest RMSE (37.549), and MAE (22.183), indicating strong accuracy and precision. Bi-LSTM closely followed with an R² of 0.971 but had a slightly higher RMSE (38.519). GRU also performed well, with an R² of 0.969 and comparable metrics. XGBoost achieved a satisfactory R<sup>2</sup> of 0.961 but with higher RMSE (45.189) and MAE (25.134). MLP had the lowest overall performance, with an R² of 0.936 and significantly higher RMSE (57.665) and MAE (37.627). In this study, LSTM emerged as the most effective model for accurate air quality predictions. 

#### **Coarse particles matter (PM10)** 

Particle size contributes to the health and ecological impacts of particulate matter (PM). PM10 particles (10 micrometer or smaller) can reach the lungs upon inhalation, resulting in serious health risks to heart and lung health. Additionally, PM deposition affects ecosystems by impairing water quality and altering plant growth, especially due to the metal and organic compounds within PM<sup>47</sup> . However, XGBoost revealed some observation scatters from the identical line. As per Fig. 12a, the attained determination coefficient based on the regression scatter formula was (LSTM = 0.97, Bi-LSTM = 0.97, GRU = 0.97, MLP = 0.94, and XGBoost = 0.96). Also, Taylor diagram showed in Fig. 12b that, the observational dataset of PM10 was located at the standard deviation of 1.05. In this comparison of air quality forecasting models (Fig. 12c), Bi-LSTM achieved the highest performance with the highest R<sup>2</sup> (0.975), lowest RMSE (43.126), and MAE (24.414), indicating excellent accuracy. LSTM and GRU were closely followed, both with R<sup>2</sup> values of 0.974, but slightly higher RMSE and MAE values. XGBoost had a moderate performance, with an R<sup>2</sup> of 0.962 but a higher RMSE (53.085). MLP showed the lowest accuracy with an R² of 0.949 and the highest RMSE (61.782). Overall, Bi-LSTM proved to be the most accurate model for air quality prediction. 

#### **Ammonia (NH3)** 

Ammonia is a major contributor to nitrogen pollution. The effects of nitrogen buildup on plant species diversity and composition within impacted environments are an essential component of ammonia pollution’s impacts 

**Scientific Reports** |        (2024) 14:30957 

13 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 



<!-- Start of picture text -->
600, mn w) = ohana<br>(a) /- ) i 0201 02 9, , i RieraTM<br>500 Y7 ii 1.35 X05 %, + GRUmip<br>8 V4 i 39 0, %, XGBoost<br>a LY. H 1.20 ee,<br>| oo Wi ° i 1.05 xg 7<br>= 905975Pe ~% %ooS : & ogo} “TIS| 7 >r Xo25<br>= 500 ° BB 6 H & ry *y<br>fo) ee ae H = 075 ‘<br>a 0 ALA H E a 2.<br>200 ORTS Hi 32 0.60 ‘Me ‘<br>eeOse: onism i: *& 045) . \ \ 2‘<br>100 | Speen ° saad : 0.30 \ °<br>f Wa OXGBoost : 0.15 & y? a t 3<br>oe i Ups \t .<br>0 100 200 300 400 $00 600 700 0.08 7 ebpereesees °<br>SO, (Actual) $6565 6 SC 6A A<br>Standard deviation<br>an) R? RMSE MAE MSE 6 WI<br>LSTM 0.950 11.648 6.463 135.686 11.648 0.961<br>Bi-LSTM 0.955 11.110 6.076 123.440 11.019 0.964<br>GRU 0.949 11.812 6.382 139.524 11.808 0.959<br>__MLP 0.928 14.044 7.938 197.231 13.263 0.938<br>XGBoost 0.916 15.214 7.020 231.461 15.213 0.933<br><!-- End of picture text -->

**Fig. 10** . Actual and forecasted values of SO2 using different AI-models; ( **a** ) Scatter plots, ( **b** ) Taylor diagram, ( **c** ) Forecasting metrics. 

on biodiversity. Ammonia can cause irritation to the skin or eyes or irritation to them when it comes into contact with it. If ammonia is inhaled, it can cause coughing, wheezing, and shortness of breath, as it can cause irritation to the respiratory system. Furthermore, ammonia inhalation may cause irritation to the throat and nose. However, XGBoost revealed some observation scatters from the identical line. According to Fig. 13a, the attained determination coefficient based on the regression scatter formula was (LSTM = 0.96, Bi-LSTM = 0.97, GRU = 0.96, MLP = 0.93, and XGBoost = 0.93). As per Fig. 13b, Taylor diagram showed that the observational dataset of NH3 was located at the standard deviation of 1.05. In this assessment of air quality forecasting models (Fig. 13c), Bi-LSTM demonstrated the highest performance with an R<sup>2</sup> of 0.971, the lowest RMSE (5.283), and the lowest MAE (2.743), indicating superior accuracy and minimal error. LSTM and GRU were followed closely, with R<sup>2</sup> values of 0.963 and 0.965, respectively, but slightly higher RMSE and MAE scores. The accuracy of MLP and XGBoost was lower, with MLP having an R<sup>2</sup> of 0.939 and the highest RMSE (7.645), while XGBoost had the lowest R<sup>2</sup> (0.933) and a high RMSE (8.036). In conclusion, Bi-LSTM was the most effective model for precise air quality predictions in this study. 

Exposure to air pollution is a global public health hazard, with a considerable body of evidence linking shortterm and long-term exposures to a range of health outcomes, including all-cause and cause-specific mortality, respiratory and cardiovascular conditions, neurodevelopmental deficiencies, and adverse pregnancy and birth out-comes<sup>44</sup> . The current research was fueled by the robustness of deep learning models that predictability performance of recent research development on machine learning establishment, deep learning revealed superior performance to the other ML models. The eight air quality parameters were chosen due to their seriousness as environmental and atmospheric indices. In this regard, the study’s unique methods for predicting air pollution provide valuable results in various circumstances. Models that are considered to be suitable and beneficial for daily prediction include LSTM, Bi-LSTM, GRU, MALP and XGBoost. In order to discuss the reliability and validity of the current intelligence models, Table 3 summarizes the latest air quality studies using multiple machine and deep learning models under different meteorological conditions. 

This study’s significance lies in its model variety, particularly Bi-LSTM, which achieved higher R<sup>2</sup> values than other methods. Comparatively, previous studies achieved comparable R<sup>2</sup> values but were limited to fewer pollutants or simpler seasonal analyses. In addition, studies in Hyderabad (2022) and Kolkata (2019–2022) showed strong results with models such as ANN and XGBoost, but focused on fewer pollutants and seasonal 

**Scientific Reports** |        (2024) 14:30957 

14 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 



<!-- Start of picture text -->
1809, on oo<br>(a) / 00 01 02 4. oe<br>=~ Cy) ii 135 QX05 %, + mi<br>Fae . ok 42 6%, XGBoost<br>> 1200 6 OO IAL) i 1s eo Foy<br>4 0.0 BAS i 8 te 0%<br>e Bier O° i § 0.90} “AY Ko<br>a on On Serer : g 075 yy ys<br>600 ) Q Oeaekeyo : 5° os °<br>- Feateee 4 opnism : * 0a5p) x \&<br>100 | emer tied : 0.304 S o<br>andey OXGBoostMLP i| 0.15 h , & Sa- t ¢8<br>0 i| Ze {! | e<br>0 300 600 900 1200 1500 1800 0.0% “2 96 e8 9 8 8 8 °<br>PM) 5 (Actual) esegse Standardssodeviationfm a<br>(c) R RMSE MAE MSE o WI<br>__LSTM 0.973, 37.549 | 22.183 1409.902 37.515 0.977<br>Bi-LSTM 0.971 38.519 21.352 1483.676 38.149 0.976<br>GRU 0.969 39.775 21.674 1582.067 39.053, (0.976<br>_ MLP____s0.936 57.665 37.627 | 3325.249 52.175 0.955<br>XGBoost 0.961 45.189 25.134 | 2042.088 45.189 0.968<br><!-- End of picture text -->

**Fig. 11** . Actual and forecasted values of PM2.5 using different AI-models; ( **a** ) Scatter plots, ( **b** ) Taylor diagram, ( **c** ) Forecasting metrics. 

variations. In contrast, the Delhi study utilized Bi-LSTM, LSTM, and GRU extensively across pollutants and extended the model’s applicability across multiple seasons, enhancing predictive power for dynamic air quality conditions in a complex urban area. The high performance and adaptability of the Bi-LSTM model across pollutants highlight its potential as a robust choice for multi-pollutant air quality forecasting, enabling policymakers and environmental agencies to make data-driven decisions and mitigate pollution’s impact on public health. 

### **Conclusion, limitations and future directions** 

In this study, different air quality parameters were proposed, including CO, NO, NO2, O3, SO2, PM2.5, PM10, and NH3. Datasets were collected for monitoring stations located near Delhi, India, for the duration of (25-112020/24-01-2023) with an hourly rate. For this purpose, various AI models were introduced, including Long Short-Term Memory (LSTM), Bidirectional Long-Short Term Memory (Bi-LSTM), Gated Recurrent Unit (GRU), Multilayer Perceptron (MLP), and Extreme Gradient Boosting (XGBoost). The following findings can be drawn from the current study: 

- i.  In CO forecasting, the AI models showed the following accuracy: Bi-LSTM = 0.979, GRU = 0.977, LSTM = 0.971, XGBoost = 0.963, and MLP = 0.941, respectively. 

- ii.  In NO forecasting, the AI models showed the following accuracy: Bi-LSTM = 0.961, GRU = 0.956, LSTM = 0.948, XGBoost = 0.943, and MLP = 0.913, respectively. 

- iii.  In NO2 forecasting, the AI models showed the following accuracy: Bi-LSTM = 0.956, GRU = 0.953, LSTM = 0.946, XGBoost = 0.940 and MLP = 0.937, respectively. 

- iv.  In O3 forecasting, the AI models showed the following accuracy: GRU = 0.9624, XGBoost = 0.9619, BiLSTM = 0.9588, LSTM = 0.9579 and MLP = 0.9571, respectively. 

- v.  In SO2 forecasting, the AI models showed the following accuracy: Bi-LSTM = 0.955, LSTM = 0.950, GRU = 0.949, MLP = 0.928 and XGBoost = 0.916, respectively. 

- vi.  In PM2.5 forecasting, the AI models showed the following accuracy: LSTM = 0.973, Bi-LSTM = 0.971, GRU = 0.969, XGBoost = 0.961 and MLP = 0.936, respectively. 

- vii.  In PM10 forecasting, the AI models showed the following accuracy: Bi-LSTM = 0.9751, LSTM = 0.9744, GRU = 0.9737, XGBoost = 0.9622 and MLP = 0.9488, respectively. 

**Scientific Reports** |        (2024) 14:30957 

15 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 



<!-- Start of picture text -->
Pn » oo<br>soo | (8) o | 00 01 a2 9, 7 ra<br>1500 oreofoOGo i|| 1.35re aed Og 25 OL%& * MUPGRUXGBoost<br>F 1200 ee Spece  DOi 5 SfnaIzy 2,Te<br>= ov EDX) S> i 2% 0.90; sok,ae Xo0%*<br>s De Ec : : sx<br>*<br>600 | °@QpeaaeeaersSEE H: 28 0.60}. ‘s ‘. °°<br>ieee coat ° earisms : * 04s x s<br>300 | Deyfae O GRUa H 0.3041| . 2<br>ay OXGBoost | 0.15 h., § Y 4 7<br>o : a }1| | .<br>0 $00 1000 1500 2000 00a 2 he ne Mo he<br>PM,) (Actual) géegaReseraggagse¢5658Standard deviation35 548a<br>(c) R RMSE MAE MSE o WI<br>__LSTM 0.974 43.699 26.044 = 1909.585 43.697 0.978<br>Bi-LSTM 0.975 43.126 24.414 — 1859.862 42.961 0.979<br>GRU 0.974 44.287 28.004 1961.331 43.121 (0.978<br>__MLP 0.949 61.782 38.991 | 3817.006 60.447 0.960<br>XGBoost 0.962 53.085 29.727 2817.991 53.084 0.969<br><!-- End of picture text -->

**Fig. 12** . Actual and forecasted values of PM10 using different AI-models; ( **a** ) Scatter plots, ( **b** ) Taylor diagram, ( **c** ) Forecasting metrics. 

viii.  In NH3 forecasting, the AI models showed the following accuracy: Bi-LSTM = 0.971, GRU = 0.965, LSTM = 0.963, MLP = 0.939 and XGBoost = 0.933, respectively. 

This study highlights the potential of AI-based models in air quality forecasting, essential for proactive urban pollution management. Accurate predictions support real-time alert systems, preventing public health as cities expand. While high model accuracy was achieved, limitations remain, such as reliance on pollutant data alone, lacking real-time adaptability to sudden changes in pollution. Future research should incorporate meteorological and socioeconomic data, enabling more robust, responsive models. Real-time monitoring and long-term trend analyses could further enhance air quality management, enabling sustainable policy decisions. Interdisciplinary collaboration will be crucial in advancing these models for efficient, data-driven air quality solutions. 

**Scientific Reports** |        (2024) 14:30957 

16 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 



<!-- Start of picture text -->
350 poorer eon<br>300 H: 135)| Og 25 %, c3 GRUMP.<br>250 f i 10h 2, } 26 My, —<br>Oe ad - are<br>> Q SS ae i 105; ‘ 4 ey<br>4 200 re) ° $4 H s nae . 2, %<br>ey ope | % 0.90 (iP, xe<br>z 150 9 Os ‘- ° i aL ya Sx °<br>6 Onkeree”o i 2 osof// // » v<br>eer i & 4<br>$0 eer aD O on i 030, aes °<br>TO AE OXGBoost | os, ey # i—\ 7 8<br>0 50 100 150 200 250 300 350 / 8 4 8 EA 8 2 A 8 g 8 °<br>NH; (Actual) $565 5538 448 5<br>Standard deviation<br>__ © | R’_ RMSE MAPE | MAE MSE 6 WI<br>LSTM 0.963 5.957 18.375 3.387 35.485 5.863 0.971<br>Bi-LSTM _ 0.971 5.283 13.197 2.743 27.905 5.278 0.976<br>GRU 0.965 5.843 16.349 3.221 34.136 5.840 0.970<br>| XGBoostMLP 0.9390.933 7.6458.036 23.66913.773 4.2283.438 64.57658.445 7.5568.024 0.9540.945<br><!-- End of picture text -->

**Fig. 13** . Actual and forecasted values of NH3 using different AI-models; ( **a** ) Scatter plots, ( **b** ) Taylor diagram, ( **c** ) Forecasting metrics. 

|**Reference**|**Study area**|**Air pollutant**|**ML/DL model**|**Highlights**|
|---|---|---|---|---|
|6|Chennai, using the historical data<br>available from 2017 to 2022|AQI|XGBoost, RF, Bagging<br>Regressor, and<br>XGBRegressor|XGBoost achieved R<sup>2</sup> =0.9935, MAE=0.02, MSE=0.001, and RMSE=0.04|
|8|Ahmedabad, January 2015, to<br>January 2021|AQI|SARIMA, SVM and LSTM|LSTM showed R<sup>2</sup> =0.951, MSE=656.623 and RMSE=25.625.|
|18|Visakhapatnam, Andhra Pradesh,<br>from July 2017 to September 2022|AQI|LightGBM, RF, Catboost,<br>Adaboost, and XGBoost|Catboost model yielded (R<sup>2</sup> =0.9998) and (RMSE=0.76).|
|19|New Delhi, 2008 to 2010|PM2.5|GRU-ED, RF, XGBoost,<br>ANNs, and LSTM|GRU-ED showed (R<sup>2</sup> =0.959, NSE=0.953, MAE=1.770, RRMSE=0.002, and<br>MAPE=0.190).|
|22|Hyderabad using five-year<br>(2018–2022) data|PM2.5, NH3,<br>CO, O3, NO2,<br>and SO2|MLR, SVR, RF, ANN,<br>XGBoost, and LSTM|ANN>RF>SVR>XG Boost>LSTM>MLR. ANN outperformed other model<br>(R<sup>2</sup> >0.90) in both summer and winter seasons.|
|23|India using regional and temporal<br>data in 2018|PM2.5and<br>PM10|LightGBM|PM2.5showed R<sup>2</sup> =0.79, 0.80, 0.86 and 0.87, for original, spatial, temporal and<br>spatial-temporal, respectively. PM10showed R<sup>2</sup> =0.81, 0.82, 0.87 and 0.88, for<br>original, spatial, temporal and spatial-temporal, respectively.|
|25|New Delhi, hourly observations for<br>2021 and 2022|O3and NO2|USS-GNN|R<sup>2</sup> =0.650 and 0.618, RMSE=13.950 and 16.120, MAE=10.730 and 12.930 for<br>O3and NO2, respectively.|
|48|Kolkata from 2019 to 2022 (4 years)|PM2.5, PM10,<br>NO2, NH3, SO2<br>and CO|RF, DT, KNN, SVR, Ridge,<br>Lasso, and XGBoost|XGBoost showed R<sup>2</sup> =0.858 (Winter), R<sup>2</sup> =0.919 (Pre- Monsoon), R<sup>2</sup> =0.844<br>(Monsoon) and R<sup>2</sup> =0.862 (Post Monsoon).|
|49|Hyderabad, from January 2018 to<br>December 2019|PM2.5|MLR, DT, KNN, RF,<br>XGBoost and LSTM|XGBoost: R<sup>2</sup> =0.82 and MAE=7.01 µg/ m<sup>3</sup>. LSTM: R<sup>2</sup> =0.89 and<br>MAE=5.78 µg/ m<sup>3</sup>.|
|Current|Delhi, from 25/11/2020 to<br>24/01/2023|CO, NO, NO2,<br>O3, SO2, PM2.5,<br>PM10, and NH3|LSTM, Bi-LSTM, GRU,<br>MLP, and XGBoost|Bi-LSTM model was the best mode CO with (R<sup>2</sup> =0.979), NO with (R<sup>2</sup> =0.961),<br>NO2with (R<sup>2</sup> =0.956), SO2with (R<sup>2</sup> =0.955), PM10with (R<sup>2</sup> =0.9751) and NH3<br>with (R<sup>2</sup> =0.971). Meanwhile, GRU and LSTM models performed better in<br>forecasting O3and PM2.5with (R<sup>2</sup> =0.9624) and (R<sup>2</sup> =0.973), respectively.|



**Table 3** . Summary of previous studies on air pollutant prediction using intelligent learning approaches across India. 

**Scientific Reports** |        (2024) 14:30957 

17 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 

### **Data availability** 

The data can be shared upon request from the corresponding author. 

Received: 26 August 2024; Accepted: 2 December 2024 



### **References** 

1. Perera, F. Pollution from fossil-fuel combustion is the leading environmental threat to global pediatric health and equity: solutions exist. _Int. J. Environ. Res. Public Health_ . **15** , 16 (2018). 

2. Manisalidis, I., Stavropoulou, E., Stavropoulos, A. & Bezirtzoglou, E. Environmental and health impacts of air pollution: a review. _Front. Public. Health_ . **8** , 14 (2020). 

3. Ravindra, K. Emission of black carbon from rural households kitchens and assessment of lifetime excess cancer risk in villages of North India. _Environ. Int._ **122** , 201–212 (2019). 

4. Ravindra, K., Singh, T., Pandey, V. & Mor, S. Air pollution trend in Chandigarh city situated in Indo-Gangetic Plains: understanding seasonality and impact of mitigation strategies. _Sci. Total Environ._ **729** , 138717 (2020). 

5. Villanueva, F. et al. Ambient levels of volatile organic compounds and criteria pollutants in the most industrialized area of central Iberian Peninsula: intercomparison with an urban site. _Environ. Technol._ **37** , 983–996 (2016). 

6. Ravindiran, G. et al. Impact of air pollutants on climate change and prediction of air quality index using machine learning models. _Environ. Res._ **239** , 117354 (2023). 

7. Bodor, Z., Bodor, K., Keresztesi, Á. & Szép, R. Major air pollutants seasonal variation analysis and long-range transport of PM 10 in an urban environment with specific climate condition in Transylvania (Romania). _Environ. Sci. Pollut. Res._ **27** , 38181–38199 (2020). 

8. Maltare, N. N. & Vahora, S. Air Quality Index prediction using machine learning for Ahmedabad city. _Digit. Chem. Eng._ **7** , 100093 (2023). 

9. Yadav, V., Yadav, A. K., Singh, V. & Singh, T. Artificial neural network an innovative approach in air pollutant prediction for environmental applications: a review. _Results Eng._ , **102305** (2024). 

10. Kumar, K. & Pande, B. Air pollution prediction with machine learning: a case study of Indian cities. _Int. J. Environ. Sci. Technol._ **20** , 5333–5348 (2023). 

11. Guo, Q., He, Z. & Wang, Z. The characteristics of Air Quality changes in Hohhot City in China and their relationship with Meteorological and Socio-economic factors. _Aerosol Air Qual. Res._ **24** , 230274 (2024). 

12. Gupta, N. S. et al. Prediction of air quality index using machine learning techniques: a comparative analysis. _J. Environ. Public Health_ **2023** (2023). 

13. Nath, P., Saha, P., Middya, A. I. & Roy, S. Long-term time-series pollution forecast using statistical and deep learning methods. _Neural Comput. Appl._ , 1–20 (2021). 

14. Aggarwal, A. & Toshniwal, D. A hybrid deep learning framework for urban air quality forecasting. _J. Clean. Prod._ **329** , 129660 (2021). 

15. Janarthanan, R., Partheeban, P., Somasundaram, K. & Elamparithi, P. N. A deep learning approach for prediction of air quality index in a metropolitan city. _Sustainable Cities Soc._ **67** , 102720 (2021). 

16. Pruthi, D. & Liu, Y. Low-cost nature-inspired deep learning system for PM2. 5 forecast over Delhi, India. _Environ. Int._ **166** , 107373 (2022). 

17. Sarkar, N., Gupta, R., Keserwani, P. K. & Govil, M. C. Air Quality Index prediction using an effective hybrid deep learning model. _Environ. Pollut._ **315** , 120404 (2022). 

18. Ravindiran, G., Hayder, G., Kanagarathinam, K., Alagumalai, A. & Sonne, C. Air quality prediction by machine learning models: a predictive study on the Indian coastal city of Visakhapatnam. _Chemosphere_ **338** , 139518 (2023). 

19. Shakya, D., Deshpande, V., Goyal, M. K. & Agarwal, M. PM2.5 air pollution prediction through deep learning using meteorological, vehicular, and emission data: a case study of New Delhi, India. _J. Clean. Prod._ **427** , 139278 (2023). 

20. Dhandapani, A., Iqbal, J. & Kumar, R. N. Application of machine learning (individual vs stacking) models on MERRA-2 data to predict surface PM2.5 concentrations over India. _Chemosphere_ **340** , 139966 (2023). 

21. Ravindra, K. et al. Application of machine learning approaches to predict the impact of ambient air pollution on outpatient visits for acute respiratory infections. _Sci. Total Environ._ **858** , 159509 (2023). 

22. Suthar, G., Singh, S., Kaul, N. & Khandelwal, S. Prediction of land surface temperature using spectral indices, air pollutants, and urbanization parameters for Hyderabad City of India using six machine learning approaches. _Remote Sens. Appl. Soc. Environ._ , **101265** (2024). 

23. Wang, S. et al. Extracting regional and temporal features to improve machine learning for hourly air pollutants in urban India. _Atmos. Environ._ **338** , 120834 (2024). 

24. Rahaman, S., Tu, X., Ahmad, K. & Qadeer, A. A real-time assessment of hazardous atmospheric pollutants across cities in China and India. _J. Hazard. Mater._ **479** , 135711 (2024). 

25. Mandal, S., Boppani, S., Dasari, V. & Thakur, M. A bivariate simultaneous pollutant forecasting approach by Unified SpectroSpatial Graph Neural Network (USSGNN) and its application in prediction of O3 and NO2 for New Delhi, India. _Sustainable Cities Soc._ **114** , 105741 (2024). 

26. Guo, Q., He, Z. & Wang, Z. Monthly climate prediction using deep convolutional neural network and long short-term memory. _Sci. Rep._ **14** , 17748 (2024). 

27. Haykin, S. _Neural Networks: A Comprehensive Foundation_ (Prentice Hall PTR, 1998). 

28. Jamei, M. et al. Air quality monitoring based on chemical and meteorological drivers: application of a novel data filtering-based hybridized deep learning model. _J. Clean. Prod._ **374** , 134011 (2022). 

29. Ehteram, M., Salih, S. Q. & Yaseen, Z. M. Efficiency evaluation of reverse osmosis desalination plant using hybridized multilayer perceptron with particle swarm optimization. _Environ. Sci. Pollut. Res._ **27** , 15278–15291 (2020). 

30. Tur, R. & Yontem, S. A comparison of soft computing methods for the prediction of wave height parameters. _Knowl.-Based Eng. Sci._ **2** , 31–46 (2021). 

31. Schmidhuber, J. & Hochreiter, S. Long short-term memory. _Neural Comput._ **9** , 1735–1780 (1997). 

32. Schuster, M. & Paliwal, K. K. bidirectional recurrent neural networks. _IEEE Trans. Signal Process._ **45** , 2673–2681 (1997). 

33. Yaseen, Z. M. et al. Development of advanced data-intelligence models for radial gate discharge coefficient prediction: modeling different flow scenarios. _Water Resour. Manage_ . **37** , 5677–5705 (2023). 

34. Wang, S. et al. Air pollution prediction via graph attention network and gated recurrent unit. _Computers Mater. Continua_ . **73** , 673–687 (2022). 

35. Chen, T. & Guestrin, C. Xgboost: a scalable tree boosting system. In _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ Vol. 2016, 785–794 (ACM, 2016). 

36. Fan, J. et al. Evaluation of SVM, ELM and four tree-based ensemble models for predicting daily reference evapotranspiration using limited meteorological data in different climates of China. _Agric. For. Meteorol._ **263** , 225–241 (2018). 

**Scientific Reports** |        (2024) 14:30957 

18 

| https://doi.org/10.1038/s41598-024-82117-z 

www.nature.com/scientificreports/ 

37. Zhang, W., Wu, C., Zhong, H., Li, Y. & Wang, L. Prediction of undrained shear strength using extreme gradient boosting and random forest based on bayesian optimization. _Geosci. Front._ **12** , 469–477 (2021). 

38. Ferreira, L. B. & da Cunha, F. F. New approach to estimate daily reference evapotranspiration based on hourly temperature and relative humidity using machine learning and deep learning. _Agric. Water Manage._ **234** , 106113 (2020). 

39. Ni, L. et al. Streamflow forecasting using extreme gradient boosting model coupled with gaussian mixture model. _J. Hydrol._ **586** , 124901 (2020). 

40. Xu, C. et al. A study of predicting irradiation-induced transition temperature shift for RPV steels with XGBoost modeling. _Nuclear Eng. Technol._ **53** , 2610–2615 (2021). 

41. Feigl, M., Lebiedzinski, K., Herrnegger, M. & Schulz, K. Machine learning methods for stream water temperature prediction. _Hydrol. Earth Syst. Sci. Dis._ **2021** , 1–35 (2021). 

42. Sikorska-Senoner, A. E. & Quilty, J. M. A novel ensemble-based conceptual-data-driven approach for improved streamflow simulations. _Environ. Model. Softw._ **143** , 105094 (2021). 

43. Tao, H. et al. Development of new computational machine learning models for longitudinal dispersion coefficient determination: case study of natural streams, United States. _Environ. Sci. Pollut. Res._ **29** , 35841–35861 (2022). 

44. de Bont, J. et al. Ambient air pollution and daily mortality in ten cities of India: a causal modelling study. _Lancet Planet. Health_ . **8** , e433–e440 (2024). 

45. Guo, Q., He, Z. & Wang, Z. Predicting of daily PM2.5 concentration employing wavelet artificial neural networks based on meteorological elements in Shanghai, China. _Toxics_ **11** , 51 (2023). 

46. Guo, Q., He, Z. & Wang, Z. Simulating daily PM2.5 concentrations using wavelet analysis and artificial neural network with remote sensing and surface observation data. _Chemosphere_ **340** , 139886 (2023). 

47. Guo, Q., He, Z. & Wang, Z. Prediction of hourly PM2.5 and PM10 concentrations in Chongqing City in China based on artificial neural network. _Aerosol Air Qual. Res._ **23** , 220448 (2023). 

48. Mondal, S., Adhikary, A. S., Dutta, A., Bhardwaj, R. & Dey, S. Utilizing machine learning for air pollution prediction, comprehensive impact assessment, and effective solutions in Kolkata, India. _Results Earth Sci._ **2** , 100030 (2024). 

49. Gokul, P., Mathew, A., Bhosale, A. & Nair, A. T. Spatio-temporal air quality analysis and PM2.5 prediction over Hyderabad City, India using artificial intelligence techniques. _Ecol. Inf._ **76** , 102067 (2023). 

### **Acknowledgements** 

Ali Alsuwaiyan and Zaher Mundher Yaseen would like to thank King Fahd University of Petroleum & Minerals, Saudi Arabia for its support. 

### **Author contributions** 

Omer A. Alawi: Conceptualization, Methodology, Validation, Formal analysis, Investigation, Data Curation, Writing - Original Draft, Writing - Review & Editing. Haslinda Mohamed Kamar: Formal analysis, Resources, Supervision, Project administration, Funding acquisition. Ali Saleh Mohammed AlSuwaiyan: Methodology, Software, Formal analysis, Writing - Original Draft. Zaher Mundher Yaseen: Methodology, Validation, Formal analysis, Investigation, Writing - Original Draft, Writing - Review & Editing, Visualization. 

### **Declarations** 

### **Competing interests** 

The authors declare no competing interests. 

### **Additional information** 

**Correspondence** and requests for materials should be addressed to Z.M.Y. 

**Reprints and permissions information** is available at www.nature.com/reprints. 

**Publisher’s note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations. 

**Open Access** This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit  h t t p : / / c r e a t i v e c o m m o n s . o r g / l i c e n s e s / b y - n c - n d / 4 . 0 / . 

© The Author(s) 2024 

**Scientific Reports** |        (2024) 14:30957 

19 

| https://doi.org/10.1038/s41598-024-82117-z 

