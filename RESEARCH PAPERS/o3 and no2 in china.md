Full length article
Forecasting O3 and NO2 concentrations with spatiotemporally continuous
coverage in southeastern China using a Machine learning approach
Zeyue Li a, Jianzhao Bi b, Yang Liu c, Xuefei Hu a,*
a School of Geospatial Engineering and Science, Sun Yat-sen University, Zhuhai 519082, China
b Department of Environmental & Occupational Health Science, University of Washington, Seattle, WA 98105, USA
c Gangarosa Department of Environmental Health, Rollins School of Public Health, Emory University, Atlanta, GA 30322, USA
A R T I C L E I N F O
Handling Editor: Dr. Xavier Querol
```
Keywords:
```
Ozone
Nitrogen dioxide
Air Pollution Forecast
Chemical Transport Model
and Random Forest
A B S T R A C T
```
Ozone (O3) is a significant contributor to air pollution and the main constituent of photochemical smog that
```
```
plagues China. Nitrogen dioxide (NO2) is a significant air pollutant and a critical trace gas in the Earth’s at-
```
mosphere. The presence of O3 and NO2 has detrimental effects on human health, the ecosystem, and agricultural
production. Forecasting accurate ambient O3 and NO2 concentrations with full spatiotemporal coverage is
pivotal for decision-makers to develop effective mitigation strategies and prevent harmful public exposure.
```
Existing methods, including chemical transport models (CTMs) and time series at air monitoring sites, forecast O3
```
and NO2 concentrations either with nontrivial uncertainty or without spatiotemporally continuous coverage. In
this research, we adopted a forecasting model that integrates the random forest algorithm with NASA’s Goddard
```
Earth Observing System “Composing Forecasting” (GEOS-CF) product. This approach offers spatiotemporally
```
continuous forecasts of O3 and NO2 concentrations across southeastern China for up to five days in advance. Both
overall validation and spatial cross-validation revealed that our forecast framework significantly surpassed the
initial GEOS-CF model for all validation metrics, substantially reducing the errors in the GEOS-CF forecast data.
Our model could provide accurate near-real-time O3 and NO2 forecasts with continuous spatiotemporal coverage.
1. Introduction
```
Ozone (O3) is a significant contributor to air pollution and the pri-
```
```
mary constituent of photochemical smog that plagues China (Wang et al.
```
```
2020). Nitrogen dioxide (NO2) is a crucial trace gas and a major air
```
pollutant in the atmosphere, and it is the pivotal precursor to secondary
```
air pollutants like O3 and PM2.5 (Wang et al. 2020). The presence of
```
ground O3 and NO2 has damaging effects on human health, affecting
```
respiratory (Anenberg et al. 2022; Jerrett et al. 2009), cardiovascular
```
```
(Cakmak et al. 2016; Hoffmann et al. 2012; Wang et al. 2023), prema-
```
```
ture death (Khomenko et al. 2021), and nervous systems (Martínez-
```
```
Lazcano et al. 2013). Exposure to ambient O3 was estimated to
```
```
contribute to around 80,000 premature deaths annually (Feng et al.
```
```
2019), while NO2 has emerged as a significant air pollutant in China,
```
with an estimated 139,437 deaths linked to NO2 pollution in 2019, the
```
largest in the world (Song et al. 2023). In addition, ozone also has a
```
negative impact on the ecosystem and agricultural production, including
```
impeding plant growth (Ainsworth et al. 2012), accelerating leaf
```
```
senescence (Yendrek et al. 2013), and reducing crop output (Rai and
```
```
Agrawal 2012). Likewise, NO2 is inversely correlated with greenness
```
across different regions and seasons and has negative effects on crop
```
growth worldwide (Lobell et al. 2022). Thus, it is critical to obtain
```
precise near-term forecasts of O3 and NO2 concentrations so that envi-
ronmental agencies can be alerted to potential pollution episodes and
further develop effective mitigation strategies.
In order to numerically forecast the spatiotemporal O3 and NO2
concentrations from the next few hours to days, the use of chemical
```
transport models (CTMs) has proliferated due to their ability to simulate
```
```
complex chemical processes (Hou et al. 2022; Sun et al. 2021). Common
```
CTM forecast data consist of forecasts obtained from global models such
```
as the National Aeronautics and Space Administration’s (NASA) God-
```
```
dard Earth Observing System Composition Forecast (GEOS-CF) (Keller
```
```
et al. 2021) and the Copernicus Atmosphere Monitoring Service (CAMS)
```
```
(Casciaro et al. 2022), as well as regional models like the Comprehensive
```
```
Air Quality Model with Extensions (CAMx) (Liu et al. 2018), the Com-
```
```
munity Multiscale Air Quality Modeling system (CMAQ) (Sayeed et al.
```
```
2021), and the Weather Research and Forecasting model coupled with
```
```
Chemistry (WRF-Chem) (Zhou et al. 2017). The CTMs have the
```
- Corresponding author.
```
E-mail address: huxf9@mail.sysu.edu.cn (X. Hu).
```
Contents lists available at ScienceDirect
Environment International
journal homepage: www.elsevier.com/locate/envint
```
https://doi.org/10.1016/j.envint.2024.109249
```
```
Received 4 October 2024; Received in revised form 13 December 2024; Accepted 30 December 2024
```
```
Environment International 195 (2025) 109249
```
Available online 2 January 2025
```
0160-4120/© 2025 The Author(s). Published by Elsevier Ltd. This is an open access article under the CC BY-NC-ND license (http://creativecommons.org/licenses/by-
```
```
nc-nd/4.0/).
```
capability to model diverse atmospheric and chemical phenomena,
enabling them to forecast the distribution of air pollutant concentrations
```
across different regions and times (Mo et al. 2020). However, the CTM
```
forecast products suffer from significant discrepancies owing to un-
certainties in emission inventories, model boundary and initial settings,
```
and the parameterization of complex atmospheric processes (Zhang
```
```
et al. 2020). To improve model performance, various data assimilation
```
```
methods, including the variation method (3D-Var or 4D-Var) (Tang et al.
```
```
2017), optimal interpolation (Chai et al. 2017), and the ensemble Kal-
```
```
man filter (KF) (Tang et al. 2011), have been adopted to enhance
```
boundary and initial settings or to modify emission variables. Further-
more, various techniques and datasets have been applied to minimize
```
uncertainties in emission inventories (Zheng et al. 2019). Despite the
```
```
efforts, the numerical models continue to exhibit significant bias (Sun
```
```
et al. 2021).
```
Machine learning algorithms differ inherently from the traditional
numerical models. They are developed exclusively based on past data
and do not include any knowledge about the fundamental physical
principles. Compared to CTM-based methods, machine learning algo-
rithms are generally more efficient in terms of computation, need less
input data, and offer enhanced performance regarding precision.
Currently, machine learning approaches have gained popularity as
effective techniques for forecasting O3 and NO2 concentrations. How-
ever, such efforts have been mainly focused on forecasting the time se-
ries of O3 and NO2 concentrations at individual air monitoring sites. For
```
example, Eslami et al. (2020) applied a deep convolutional neural
```
```
network (CNN) to forecast the O3 levels on an hourly basis for each day
```
using data from the preceding day. The model can generate O3 con-
centration forecasts for the following 24-hour period in under a minute.
```
AlOmar et al. (2020) proposed an artificial neural network (ANN)
```
```
coupled with wavelet transform (WT) to forecast O3 concentrations over
```
different time intervals, specifically for 2, 3, 4, and 5 h into the future.
```
Mendes et al. (2022) employed optimized regression models combining
```
```
multiple regression (MR) and Classification and Regression Tree (CART)
```
to forecast NO2 concentrations in Portugal and Macao for the next day.
```
Vasseur and Aznarte (2021) implemented quantile regression models to
```
forecast the distribution of NO2 concentrations in an urban location for
```
up to 60 h into the future. Al Yammahi and Aung (2023) used machine
```
learning models, including nonlinear autoregressive neural network
```
(NAR-NN), seasonal autoregressive integrated moving average (SAR-
```
```
IMA), long short-term memory (LSTM), and autoregressive integrated
```
```
moving average (ARIMA), to forecast NO2 levels at 14 monitoring sites
```
```
in the United Arab Emirates (UAE) for 1–31 days in advance. However,
```
O3 and NO2 time-series forecasts at monitoring sites generally cannot
cover the entire study region, as monitoring stations are unevenly
distributed, with the majority located in urban regions and the
remainder in suburban and rural areas. While these algorithms excel in
forecasting O3 and NO2 variations at individual air monitoring sites,
their usefulness is limited. Thus, accurate forecasts of O3 and NO2 var-
iations across space and time are essential.
To date, there have been a few attempts to use machine learning
models to forecast the spatiotemporal fluctuations in O3 and NO2 con-
```
centrations. For example, Sun et al. (2021) proposed a deep learning
```
framework using LSTM layers to predict PM2.5 and ozone levels up to 48
h in advance. They spatially extrapolate their forecasts to the entire
study region by applying a spatial correction approach to CMAQ simu-
```
lations. Cheng et al. (2022) introduced a hybrid VAE-GAN model to
```
comprehend the dynamic changes of ozone levels in the spatiotemporal
```
space. This model combines a generative adversarial network (GAN)
```
```
with a variational autoencoder (VAE) and employs a recursive
```
approach, using past ozone forecasts to predict future concentrations.
```
Liu et al. (2018) implemented ARIMA models coupled with numerical
```
```
forecasts derived from CMAQ and CAMx for hourly (1–72 h) and daily
```
```
(1–3 days) forecasting of O3, NO2, and PM2.5 levels at three monitoring
```
sites, each representing urban, rural, and roadside settings. However,
the current research has several drawbacks. First, the current studies
often used imprecise spatial data, such as geographic interpolation and
CTM simulations, to forecast spatial fluctuations of O3 and NO2. Expo-
sure studies have demonstrated that statistical models coupled with
ground measurements and meteorological/land use variables can yield
```
spatial distributions with greater accuracy (Chu et al. 2016). Second, the
```
multi-phase modeling procedure would result in the accumulation of
errors, increasing overall modeling uncertainty. Third, O3 and NO2
forecasts from previous steps tend to be used for next-step forecasts,
leading to the erroneous inflating of the validation performance.
In this study, we integrated a resilient machine learning model with a
readily available global CTM forecast dataset to create a near-real-time
O3 and NO2 forecast model. By incorporating ground truth, we hoped to
enhance the CTM forecasts through the implementation of the machine
learning architecture. To overcome the limitations of previous research,
```
we built a one-stage model by adopting the Random Forest (RF) algo-
```
rithm, which has been proven effective in predicting spatial variations of
```
air pollutant concentrations (Ma et al. 2021; Zhan et al. 2018). Addi-
```
tionally, we structured the training and validation phases in such a way
that the model avoids overfitting by not utilizing future O3 and NO2
observations in the training process. The proposed method can forecast
spatiotemporally continuous daily-averaged O3 and NO2 concentrations
with high accuracy for the next five days. It’s worth noting that accu-
rately forecasting air pollutant concentrations is crucial for both short-
and long-term reasons. Short-term forecasts are essential to prevent
harmful levels from being reached, while long-term forecasts are vital
for effective air resource management and understanding the potential
```
long-term effects on public health and the environment (Yuval et al.
```
```
2012).We then validated our forecast results by comparing them with
```
ground observations.
2. Materials and methods
```
We chose southeastern China (Fig. 1), including Hunan, Jiangxi,
```
Fujian, Zhejiang, and Guangdong provinces, as the study area due to the
severe O3 and NO2 pollution in this region. The study region covers the
```
Guangdong-Hong Kong-Macao Greater Bay Area (GBA), a strategically
```
important and economically prosperous region of China, characterized
by its extensive and multifaceted industrial system. This region experi-
ences significant O3 and NO2 pollution, driven by both high emissions of
O3 precursors, such as NO2, and a subtropical monsoon climate condu-
```
cive to ozone formation (Tang et al. 2020).
```
2.1. Ground O3 and NO2 observations
Hourly O3 and NO2 data from September 20, 2022, to December 25,
2023, were collected from ground monitoring stations operated by the
```
China National Environmental Monitoring Center (CNEMC, https://air.
```
```
cnemc.cn:18007/). Fig. 1 shows the locations of the monitoring stations
```
```
(total sites N = 365) in the research domain. We determined the
```
```
maximum daily 8-hour average (MDA8) O3 concentrations and the daily
```
mean NO2 concentrations for each site, using these values as the
dependent variable in our model.
2.2. O3 and NO2 forecast data
We obtained the O3 and NO2 forecast data from NASA’s Goddard
```
Earth Observing System Composition Forecast (GEOS-CF) (https://
```
```
gmao.gsfc.nasa.gov/), a publicly accessible CTM database. Currently,
```
GEOS-CF offers near-real-time, 25 km resolution, global 5-day opera-
tional forecasts of O3 and NO2 concentrations utilizing the GEOS-Chem
```
atmospheric chemistry module (https://www.geos-chem.org)
```
```
(Knowland et al. 2022). Although biases exist in GEOS-CF ozone fore-
```
casts due to uncertainties surrounding ozone production resulting from
```
the oxidation of isoprene. (Bates and Jacob 2019) and inaccuracies
```
```
related to ozone deposition to moist surfaces (Travis and Jacob 2019),
```
GEOS-CF can capture the well-established patterns of background
Z. Li et al. Environment International 195 (2025) 109249
2
```
surface ozone (Keller et al. 2021). Likewise, systematic biases also exist
```
in GEOS-CF NO2 data because NO2 analysis is complicated by interfer-
ence from other nitrogen compounds and its short atmospheric lifetime,
making it difficult to model its surface variability. Nevertheless, the
modeled NO2 values agree well with observations over Africa, North
```
America, and Asia in summer time (Keller et al. 2021). For this research,
```
we employed the surface-level two-dimensional GEOS-CF O3 and NO2
```
forecast data to compute maximum daily 8-hour average (MDA8) ozone
```
concentrations and daily mean NO2 concentrations for the subsequent
```
five days using the China Standard Time (CST), which is in the GMT + 8
```
Time Zone.
2.3. Meteorological forecast data
We obtained the GEOS-CF surface-level meteorological data across
the 5-day forecasting period. Involved parameters include total
```
precipitation (kg/m2/s), total cloud area fraction (unitless), 10-m spe-
```
```
cific humidity (kg/kg), surface skin temperature (K), 10-m air temper-
```
```
ature (K), planetary boundary layer height (m), 10-m eastward/
```
```
northward wind (m/s), tropopause pressure based on blended estimate
```
```
(Pa), and surface pressure (Pa). We computed daily mean values for the
```
meteorological data and employed these meteorological factors in our
forecast model as spatiotemporally varying predictors.
In addition, GEOS-CF offers data-driven historical estimates of global
atmospheric composition and weather conditions, utilizing meteoro-
```
logical observations to enhance the reliability of the estimates (Keller
```
```
et al. 2021). We leveraged GEOS-CF historical data to create a present-
```
day model, a crucial step in this analysis for optimizing the parameters
of our forecast model.
Fig. 1. Study area.
Z. Li et al. Environment International 195 (2025) 109249
3
2.4. Land use data
In our forecast model, we utilized land use variables as predictors
that vary spatially to account for spatial variations in O3 and NO2 levels.
The variables incorporated the LandScan ambient population in 2022 at
```
900 m resolution (https://landscan.ornl.gov), primary and secondary
```
```
road lengths obtained and calculated from the OpenStreetMap (OSM)
```
```
road network data (https://www.openstreetmap.org), and the Coper-
```
```
nicus Climate Change Services (C3S) global Land Cover (LC) products
```
```
from 2018 at ~ 300 m resolution (https://cds.climate.copernicus.eu/),
```
```
from which percentages (%) of water bodies, urban areas, and vegeta-
```
tion cover were calculated. Land use variables are aggregated within a 1
× 1 km2 square buffer centered on each monitoring site and around the
```
centroid of each GEOS-CF grid cell (Hu et al. 2014).
```
2.5. Model training and forecasting
We construct our forecast framework based on the RF algorithm. The
RF model builds an ensemble of decision trees, where each tree is grown
by using the bootstrap sampling method, and node splitting is based on
the optimal choice from a randomly selected subset of available pre-
dictors. The predictions are finally obtained by using a majority vote or
the averaged results. The algorithm also assesses the “importance” of
each predictor by measuring the increase in prediction error when the
data for that variable are shuffled while leaving all other variables un-
changed. The RF model has gained a lot of traction due to its high
prediction accuracy, robustness when handling numerous features, and
```
ease of implementation (Hu et al. 2017). For model training, two major
```
hyperparameters of the RF model need to be set, including the number of
```
predictors randomly tried at each split (mtry) and the number of decision
```
```
trees (ntree). We built a present-day prediction model, which is not the
```
forecast model and is solely used for hyperparameter tuning and pre-
dictor selection. Ground O3 and NO2 data served as the dependent
variable in the present-day model, whereas land use factors and same-
day GEOS-CF meteorological fields served as predictors. The hyper-
parameter values were selected by minimizing the model’s out-of-bag
```
(OOB) error for the present day. Particularly, we determined ntree and
```
mtry to be 500 and 5, respectively. In addition, we selected predictors for
our forecast model based on the RF variable importance estimates,
excluding variables with importance values significantly lower than
those of others.
The forecast model was designed to preclude future data in the
training process to imitate the real-world forecast scenarios. The forecast
model was built for each individual day from the first day to the fifth day
```
on a rolling basis. The first-day (day 1) model aims to forecast NO2 and
```
```
O3 concentrations the day after the current day, and the second-day (day
```
```
2) model is to forecast NO2 and O3 concentrations two days after the
```
```
current day. Likewise, the third- (day 3), fourth- (day 4), and fifth-day
```
```
(day 5) models target the third, fourth, and fifth days after the current
```
```
day, respectively. Several rolling periods suggested by Bi et al. (2022),
```
including 10, 30, 60, and 90 days, were tested to determine the ideal
period in order to yield the best forecast accuracy. For model training,
considering the first-day model with a rolling period of 10 days, we used
air pollutant observations from day   9 to day 0 as the dependent vari-
able and the GEOS-CF air pollutant and meteorological first-day fore-
casts from day   10 to day   1 as predictors. For model forecasting, we
```
employed GEOS-CF air pollutant and meteorological first-day (day 1)
```
forecasts on day 0 to forecast NO2 and O3 concentrations on day 1. By
comparing model performance, we set the rolling window at 60 days
```
(Tables S1 and S2). The detailed strategy for constructing the forecasting
```
model for each day is summarized in Table 1.
We created a convolutional layer as an extra spatiotemporal pre-
dictor by interpolating the present-day measurements using ordinary
kriging for O3 and NO2, respectively. It is important to note that the term
’convolutional layer’ in this context differs from its usage in traditional
```
deep convolutional neural networks (CNNs). Unlike the neural network
```
architecture of CNNs, our convolutional layer refers to a pre-generated
two-dimensional surface of pollutant concentrations, which is used as
a predictor in the model prior to the modeling stage. The convolutional
layer enabled the forecast models to account for spatiotemporal auto-
correlation of O3 and NO2 concentrations among nearby observations as
well as between the current day and the forecast days.
2.6. Model validation
We evaluated our forecast models by comparing our O3 and NO2
forecasts with ground observations. Our validation was out-of-sample
since the training procedure did not contain ground observations on
the forecast days. We first generated an out-of-sample dataset for vali-
dation. Using the dataset, we employed three validation approaches,
including an overall validation, a site-specific validation, and a day-
specific validation. The overall validation utilizes all available valida-
tion samples throughout the complete modeling timeframe to provide an
accurate assessment of the model’s overall forecasting capabilities and
performance. The site-specific validation was carried out separately for
each monitoring location, focusing on evaluating the model’s effec-
tiveness in capturing the temporal variability of pollutant concentra-
tions at each monitoring site. The day-specific validation was performed
on a daily basis, with the primary objective of assessing the model’s
capability to precisely account for the spatial variations in pollutant
concentrations within a single day. We employed four validation met-
```
rics, comprising the out-of-sample coefficient of determination (R2),
```
```
normalized mean bias (NMB), mean absolute percentage error (MAPE),
```
```
and root-mean-square error (RMSE). Furthermore, a 10-fold spatial
```
```
cross validation (CV) was performed to assess the accuracy of our
```
forecasts in areas lacking ground monitors. The ground monitors were
first randomly split into 10 subsets of nearly identical size. In every
iteration of CV, one subset was withheld as the test set, where the
pollutant measurements were not used in the forecast modeling or the
computation of pollutant convolutional layers, while another nine sub-
sets were used for model training. Ten iterations of this approach were
conducted until each subset was tested.
Table 1
```
The Method to Match Ground Measurements with the Convolutional Layer and GEOS-CF Forecast Data (Air Pollutant and Meteorology Forecasts) in the Forecast Model
```
Training and Forecasting Processa.
Forecast Day Day of the Convolutional Layer
Training Prediction
CTM Running Day Pollutant Observations and CTM Forecast Day CTM Running Day CTM Forecast Day
Day 1
Day 0
Day   N to   1
```
Day   (N-1) to 0 Day 0
```
Day 1
```
Day 2 Day   (N + 1) to   2 Day 2
```
```
Day 3 Day   (N + 2) to   3 Day 3
```
```
Day 4 Day   (N + 3) to   4 Day 4
```
```
Day 5 Day   (N + 4) to   5 Day 5
```
```
a N is the rolling period (N = 60 days). Day 0 is the present day, Day 1 is the next day, etc.
```
Z. Li et al. Environment International 195 (2025) 109249
4
3. Results
3.1. Overall and spatial CV validation
Table 2 displays the overall model effectiveness of the 5-day O3 and
NO2 forecast across the one-year period from December 20, 2022, to
December 19, 2023. The results showed that our forecast model
consistently surpassed the GEOS-CF model during the 5-day forecasting
period, with all validation metrics substantially improved. In addition,
the results also showed that from the first to the fifth forecasting day, R2
decreased while RMSE and MAPE increased, indicating deteriorating
model performance as the forecasting period extended. Furthermore,
the model performs better for NO2 than for O3, with higher R2 values and
lower RMSE values across all five forecasting days. In addition, a
comparative analysis of the overall validation performance of our model
outputs and the original GEOS-CF forecasts during the study period is
presented in Tables S3 and S4, categorized by concentration levels based
on China’s ambient air pollution standards. Specifically, the concen-
```
tration thresholds for O3 were set at 100 μg/m3 (level 1) and 160 μg/m3
```
```
(level 2), while those for NO2 were set at 40 μg/m3 (level 1) and 80 μg/
```
```
m3 (level 2). The validation results indicate that both our forecast model
```
and the GEOS-CF model exhibit decreased performance at higher O3 and
NO2 concentration levels. However, our forecast model consistently
outperformed the original GEOS-CF model across all concentration
levels, as demonstrated by a range of comprehensive validation metrics.
Fig. 2 presents a comparative visualization of O3 and NO2 ground
```
observations against forecasts generated by (a) the GEOS-CF model and
```
```
(b) our proposed forecast model, highlighting their respective perfor-
```
mance in replicating real-world O3 and NO2 concentrations. The results
showed that our forecast model achieved a marked improvement over
the original GEOS-CF model, producing significantly more accurate
forecasts as measured by key validation metrics. The visual comparison
revealed that our model’s predictions were more strongly aligned with
the actual values, as evidenced by a regression line that closely
approximated the ideal 1:1 relationship, indicating a higher degree of
accuracy. Despite a lack of strong agreement between GEOS-CF O3 and
NO2 forecasts and ground observations, moderate linear correlations
were observed between them for all five forecast days, with correlation
coefficients consistently exceeding 0.48 and 0.45 for O3 and NO2,
respectively, highlighting the presence of systematic biases in the orig-
inal GEOS-CF O3 and NO2 forecasting results.
Table 3 shows the 10-fold spatial CV of the 5-day O3 and NO2 fore-
casts across the one-year period from December 20, 2022, to December
19, 2023. The modeling data set used for the overall validation was also
used for the spatial CV, with one subset withheld for testing and nine
subsets used for training in each round of the CV. The results showed
that the spatial CV results were broadly consistent with the overall
validation, showing similar performance trends. The most notable dif-
ferences were marginally reduced R2 and a modest increase in both
RMSE and MAPE values. NMB remained comparable. The spatial CV
results also substantially outperformed the validation results of the
initial GEOS-CF model across the 5-day forecasting period. The spatial
CV results also demonstrate a better model performance for NO2 than for
O3.
Our analysis confirmed that the initial GEOS-CF O3 and NO2 forecast
data exhibited substantial biases, consistent with previous evaluation
```
studies (Keller et al. 2021). By building an association between ground
```
observations and the GEOS-CF forecast data, our forecast model effec-
tively functions as a statistical correction technique for the initial GEOS-
CF forecasts and is able to provide substantially more reliable O3 and
NO2 forecasts with continuous spatial coverage.
3.2. Stie- and Day-Specific validation
Fig. 3 shows the site- and day-specific evaluation results for O3 and
NO2 over the five forecast days. The results showed that for both pol-
lutants, our RF-based forecast model achieved a substantial increase in
forecast accuracy when compared to the GEOS-CF model.
Specifically, for both pollutants, our forecast framework performed
significantly better than the initial GEOS-CF model for the site-specific
evaluation with higher R2, reduced RMSE and MAPE, and closer to
zero NMB values. For instance, the median R2 values of our O3 fore-
casting model over the five forecasting days range from 0.40 to 0.62,
while those of the GEOS-CF model range from 0.16 to 0.25. The gaps
between our NO2 forecasting model and the GEOS-CF model are even
wider. The median R2 values for our NO2 forecasting model across a five-
day forecast period vary between 0.50 and 0.63. In contrast, the GEOS-
CF model’s median R2 values are consistently negative. In addition, for
all five forecast days, compared to the GEOS-CF model, our model
```
exhibited narrower interquartile ranges (IQR) of R2 for both O3 and NO2
```
```
(Tables S5 and S6), highlighting its improved robustness.
```
For the day-specific validation, our forecast framework again per-
formed better than the initial GEOS-CF model across all validation
metrics for both O3 and NO2, including higher R2, lower RMSE and
MAPE, and closer to zero NMB values. In addition, for O3, our forecast
model generated similar IQRs of R2 for the first three forecast days and
narrower IQRs of R2 for the last two forecast days compared to those of
```
the GEOS-CF model (Table S7). For NO2, the IQRs of R2 for all five
```
```
forecasting days are narrower (Table S8). Note that the dispersion of R2
```
values was more pronounced in the day-specific validation, with IQRs
wider than those in the site-specific validation for both pollutants,
suggesting that our model faces more difficulty forecasting the spatial
variability of NO2 and O3 levels in a specific day than the temporal
variability for a particular site, a result consistent with previous research
```
(Bi et al. 2022). Temporal variability in pollutant concentrations at
```
monitoring sites can be largely explained by the periodic patterns of
```
weather conditions and emissions (Xu and Zhang 2020), which our
```
model likely captures accurately through its incorporation of meteoro-
logical data and ground measurements. The site- and day-specific vali-
dation further underscores that our forecasting model excels more
significantly for NO2 than for O3. This is evidenced by the more sub-
stantial enhancements in all validation metrics when comparing our
Table 2
Comparison of Overall Validation Performance between Our Forecast Model
```
(RF + GEOS-CF) and the Initial CTM Forecast Model (GEOS-CF) for the Period
```
from December 20,2022 to December 19, 2023a.
Forecast
Day
N of the
Test
Sample
R2
RMSE
```
(μg/
```
```
m3)
```
MAPE
```
(%)
```
NMB
```
(%)
```
O3
RF + GEOS-CF
Day 1 132,421 0.6339 21.4702 27.97   1.34
Day 2 132,418 0.5542 23.7153 31.43   1.49
Day 3 132,416 0.5081 24.9129 33.47   2.06
Day 4 132,413 0.4622 26.0637 35.03   1.86
Day 5 132,412 0.4263 26.9257 36.32   2.22
GEOS-CF
Day 1 132,421 0.2020 31.7003 41.09 2.49
Day 2 132,418 0.1818 32.1287 41.73 2.06
Day 3 132,416 0.1579 32.5965 42.62 1.58
Day 4 132,413 0.1233 33.2783 43.30 1.38
Day 5 132,412 0.0667 34.3432 44.81 2.27a
The rolling period was 60 days.
NO2
RF + GEOS-CF
Day 1 135,822 0.7152 7.0351 32.27 0.83
Day 2 135,817 0.6661 7.6211 35.39 1.10
Day 3 135,813 0.6498 7.8046 36.52 1.27
Day 4 135,809 0.6338 7.9904 37.11 0.97
Day 5 135,805 0.6180 8.1882 37.97 1.00
GEOS-CF
Day 1 135,822   0.1672 14.2432 59.87   20.23
Day 2 135,817   0.1818 14.3370 60.75   20.30
Day 3 135,813   0.1915 14.3967 61.18   20.31
Day 4 135,809   0.2033 14.4848 61.54   19.89
Day 5 135,805   0.1795 14.3887 59.78   25.92a
The rolling period was 60 days.
Z. Li et al. Environment International 195 (2025) 109249
5
```
Fig. 2. Scatter plots between observed values and model forecasts for (A) O3 and (B) NO2 .
```
Z. Li et al. Environment International 195 (2025) 109249
6
NO2 forecasting model to the GEOS-CF model.
```
Fig. 4(A) and 4(B) show the fluctuations of the day-specific evalua-
```
tion MAPE values for the first forecast day and daily O3 and NO2 con-
centrations from December 20, 2022, to December 19, 2023. The results
suggest that for both pollutants, MAPE tended to rise following a sharp
drop in pollutant concentrations, indicating a diminished forecast ca-
```
pacity of our model under such conditions. Fig. 4(C) illustrates the
```
fluctuations of daily O3 and NO2 concentrations from December 20,
2022, to December 19, 2023. It’s worth noting that the timing of the
reduction in O3 and NO2 concentrations aligns closely. The simultaneous
drop in O3 and NO2 levels might be due to bursts of strong east and south
winds from the ocean that introduce cleaner air into the study area,
thereby quickly and temporarily reducing O3 and NO2 pollution. In
addition, daily O3 levels show greater variability compared to NO2,
likely due to the influx of O3 pollution from sources outside the study
area. In contrast, NO2 levels are primarily determined by local sources,
resulting in more stable patterns.
3.3. Variable importance evaluation
```
Fig. 5(A) depicts the relative importance of different variables in our
```
O3 forecast model, spanning a 5-day forecast period. For the first and
second days of forecasting, the present-day convolutional layer and the
GEOS-CF O3 forecast data emerge as the two most significant variables.
The GEOS-CF O3 forecast data remained the most important variable for
the rest of the forecast days, while the significance of the present-day
convolutional layer dropped to the third, fifth, and sixth places for the
third, fourth, and fifth forecast days, respectively. The drop in impor-
tance of the present-day ozone is expected because the longer the period
between the current day and the forecast day, the lower the correlation.
This finding revealed that the present-day convolutional layer had high
importance for short-term forecasts. The results also showed that
meteorological fields have a greater impact than land-use factors, as
reflected in their higher importance values. In addition, we found that
total precipitation, total cloud area fraction, O3 total column density,
and O3 tropospheric column density emerged as key predictors, ranking
within the top six in terms of importance for all five forecast days,
showing strong correlations with ground ozone concentrations.
```
Fig. 5(B) presents the relative importance of various variables in our
```
NO2 forecasting model, as determined for each of the five forecast days.
The present-day convolutional layer emerged as the most important
variable for all five forecast days, indicating strong correlations between
the current-day NO2 levels and future NO2 levels over the entire 5-day
forecast period. The results further showed that the GEOS-CF NO2,
NO, and NOy also contribute significantly to the model fitting process,
and they are consistently among the top four in importance ranking for
all five forecast days. Other significant variables include 10 m eastward
```
(U) wind, latitude, and longitude, which all rank relatively high in
```
importance across the 5-day forecast period. In general, meteorological
fields exert a stronger influence than land-use factors, as indicated by
their higher importance values. Land-use terms tend to be time-
invariant, limiting their ability to reflect changes in O3 and NO2 con-
centrations over time. In contrast, meteorological factors exhibit varia-
tions in both time and space, allowing them to capture not only spatial
differences but also temporal fluctuations in these pollutant concentra-
```
tions (Liu et al. 2015).
```
3.4. Spatial distributions of O3 and NO2 forecasts during special events
Fig. 6 illustrates the spatial patterns of the O3 and NO2 forecasts
generated from our forecast model and the GEOS-CF model on July 26,
2023, for the subsequent five days, during which Typhoon Doksuri made
landfall in China on July 28, 2023. In 2023, Typhoon Doksuri was the
most powerful and devastating tropical cyclone to hit China, causing
significant damage. It is considered one of the top natural disasters of
that year, impacting nearly 3 million people and resulting in economic
```
losses of approximately 14.95 billion Chinese Yuan (Zhao et al. 2024).
```
```
Fig. 6(A) showcases Doksuri’s track (green dots). It made landfall in
```
Jinjiang, Fujian Province, on July 28, 2023, and then headed north.
Three provinces of Fujian, Zhejiang, and Jiangxi were the most affected
```
areas (in red) in our study region by the storm. Fig. 6(B) depicts the O3
```
```
forecasts derived from our forecast model (a) and the GEOS-CF model
```
```
(b). Our model’s O3 forecasts for the storm-struck areas on July 28,
```
2023, show a dip to 60.60 μg/m3, lower than the preceding and
following days. GEOS-CF forecasts, however, do not show this trend,
with their lowest value of 64.77 μg/m3 occurring on July 27, 2023
```
(Table S9). This result indicates that, compared to the original GEOS-CF
```
model, our forecasting framework could better capture the sudden
```
changes in O3 levels due to special events like tropical cyclones. Fig. 6(C)
```
```
displays the NO2 forecasts from our model (a) and the GEOS-CF model
```
```
(b). Both NO2 forecasts do not show significant spatial variations during
```
the five-day forecasting period, likely attributed to the predominantly
localized and anthropogenic nature of NO2′s primary sources. In addi-
tion, we assessed our model’s ability to simulate the formation, trans-
port, and dispersion of air pollution by examining two distinct 5-day
periods, one for O3 and one for NO2, each characterized by a gradual
increase in pollutant concentrations. Fig. S1 illustrated spatial distri-
```
butions of O3 concentration forecasts from (a) our model and (b) the
```
GEOS-CF model generated on February 11, 2023, for the upcoming five
days from February 12, 2023, to February 16, 2023. The results showed
that our forecasts accurately predicted a consistent increase in O3 con-
```
centrations, ranging from 38.00 to 74.99 μg/m3 (as detailed in
```
```
Table S10), and effectively captured the spatial dynamics of O3 pollu-
```
tion, including its formation, movement, and dispersal, over the 5-day
forecasting period. In contrast, the GEOS-CF forecasts exhibited a
notable decrease in O3 concentrations on the second day of the forecast
```
period (Table S10), and the spatial pattern appeared irregular and less
```
coherent, suggesting potential limitations in capturing the complex
dynamics of O3 pollution. Fig. S2 illustrated spatial distributions of NO2
```
concentration forecasts from (a) our model and (b) the GEOS-CF model
```
initiated on May 9, 2023, for the upcoming five days from May 10, 2023,
to May 14, 2023. Our model projected a steady rise in NO2 concentra-
```
tions, increasing from 12.97 to 15.53 μg/m3 (Table S11) over the fore-
```
cast period. Conversely, the GEOS-CF forecasts exhibited fluctuations in
NO2 levels during the same period. However, the spatial distributions
predicted by both models remained relatively consistent throughout the
five-day forecast, likely due to the limited variability in NO2 concen-
trations during this period.
4. Discussion
This study adopts a novel framework utilizing the random forest
algorithm to improve the accuracy of the GEOS-CF O3 and NO2 forecasts
for the next five days at 25 km resolution. As demonstrated in this
Table 3
```
Tenfold Spatial CV Performance of Our Forecast Model (RF + GEOS-CF) for the
```
Period from December 20, 2022 to December 19, 2023a.
Forecast
Day
N of the
Test
Sample
R2
RMSE
```
(μg/
```
```
m3)
```
MAPE
```
(%)
```
NMB
```
(%)
```
Day 1 132,421 0.6318 21.5312 28.32   0.70
Day 2 132,418 0.5526 23.7570 31.55   0.94
O3 Day 3 132,416 0.5069 24.9426 33.53   1.51
Day 4 132,413 0.4604 26.1073 35.12   1.36
Day 5 132,412 0.4257 26.9413 36.36   1.79a
The rolling period was 60 days.
Day 1 135,822 0.6491 7.8051 38.59 1.08
Day 2 135,817 0.5982 8.3551 41.86 1.42
NO2 Day 3 135,813 0.5794 8.5502 43.17 1.64
Day 4 135,809 0.5645 8.7098 43.69 1.31
Day 5 135,805 0.5491 8.8932 44.49 1.28a
The rolling period was 60 days.
Z. Li et al. Environment International 195 (2025) 109249
7
investigation, the GEOS-CF O3 and NO2 forecasts exhibit discrepancies
with ground observations, yet they show moderate linear correlations
with these in-situ measurements, suggesting significant systematic bia-
ses in the original forecasts. As reported in previous studies, for O3, this
is due to uncertainties surrounding the ozone production from isoprene
```
oxidation (Bates and Jacob 2019), combined with inaccuracies in the
```
```
deposition of ozone on wet surfaces (Travis and Jacob 2019). For NO2, it
```
is because of its short atmospheric lifetime and interference from other
```
nitrogen compounds, likely contributing to the observed biases (Keller
```
```
et al. 2021). Our forecast framework yielded substantial improvements
```
in the accuracy of initial GEOS-CF O3 and NO2 forecasts, as evidenced by
comprehensive validations conducted at overall, site-specific, and day-
specific levels throughout the entire 5-day forecasting period. It
indicates that RF-based models coupled with ground observations can
generate more accurate O3 and NO2 forecasts. The spatial CV results,
mirroring the overall validation findings, underscore the reliability of
our forecast model in accurately predicting O3 and NO2 levels in regions
lacking ground-based monitoring stations. Unlike widely-used time-se-
```
ries forecast methods (Al Yammahi and Aung 2023), which focus on air
```
pollution forecasts solely at ground monitoring sites using their histor-
ical data, our RF-based framework enables a more comprehensive and
spatially extensive evaluation of air quality across the region of interest.
Overall, this study demonstrates the feasibility and necessity of recti-
fying original CTM forecasting outcomes using machine learning models
to produce accurate near-real-time O3 and NO2 forecasts. The study
further underscores that while the forecasting model demonstrates
```
Fig. 3. (a) Day-specific and (b) site-specific validation performance from December 20, 2022 to December 19, 2023 for (A) O3 and (B) NO2 .
```
Z. Li et al. Environment International 195 (2025) 109249
8
strong performance for NO2, it is less effective for O3. This discrepancy is
likely attributed to the impact of O3 pollution transported from outside
the modeled domain, which our model does not sufficiently account for.
In contrast, NO2 pollution is primarily driven by local sources due to its
shorter atmospheric lifetime, allowing for more accurate forecasts
within the model’s scope. It is also worth noting that our forecasting
framework does not provide real-time forecasts but rather operates in a
near-real-time manner. This is because our approach requires the
```
Fig. 4. Daily validation MAPE values (blue dots) with domain-average concentrations (green bars) using the first forecast day as an example for (A) O3 , (B) NO2 , and
```
```
(C) illustrates the daily fluctuations of O3 and NO2 . (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of
```
```
this article.)
```
Z. Li et al. Environment International 195 (2025) 109249
9
```
Fig. 5. RF variable importance values across the five forecast days for (A) O3 and (B) NO2 .
```
Z. Li et al. Environment International 195 (2025) 109249
10
Fig. 6. Spatial distributions of O3 and NO2 concentration forecasts on July 26, 2023 for the upcoming five days, during which Typhoon Doksuri made landfall in
```
China on July 28, 2023. (A) Doksuri’s track (green dots) and impacted areas (in red); (B) Our O3 forecasts(a) and the GEOS-CF O3 forecasts(b); (C) Our NO2 forecasts
```
```
(a) and the GEOS-CF NO2 forecasts (b). (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)
```
Z. Li et al. Environment International 195 (2025) 109249
11
collection and reporting of daily ground observations before running the
forecasting model. As a result, the forecast data is produced at the end of
the day, introducing a short delay between data acquisition and forecast
generation.
Regarding variable importance, for the O3 forecasting model, the
GEOS-CF ozone forecasts were identified as the top predictor for all five
forecast days, consistently ranking among the top two in variable
importance. It suggests that despite being biased, GEOS-CF forecasts
played a key role in explaining the variance in ozone ground observa-
tions as they offered insightful information on the spatial distribution of
O3 concentrations, which is evident by their relatively high correlations
with ground observations of ozone. In addition, the present-day ozone
convolutional layer was the top contributor to the improved model
performance, particularly for the first two forecast days, demonstrating
that the present-day ozone concentrations have a strong connection with
future ozone concentrations. It is essential for both the same-day esti-
mation and the near-future forecast, in alignment with results from prior
```
research (Bi et al. 2022; Hu et al. 2017). Total precipitation and total
```
cloud area fraction are two other important predictors, indicating their
strong impacts on ozone levels. This is because cloudy and rainy weather
```
conditions can hinder the production of ozone (Cao and Yin 2020).
```
Previous modeling studies have identified cloud fraction as a key pre-
```
dictor of ozone concentrations (Meng et al. 2022). Another two impor-
```
tant predictors are O3 total column density and O3 tropospheric column
density, as they are closely related to ground ozone concentrations. For
the NO2 forecasting model, the present-day convolutional layer emerged
as the most crucial predictor, consistently maintaining the top-ranking
position in variable importance throughout the entire 5-day fore-
casting period. This finding suggests a robust relationship between the
NO2 concentrations observed today and those forecasted for the
upcoming five days. Compared to other air pollutants like O3, NO2
```
pollution was primarily driven by emissions from local sources (Wen
```
```
et al. 2021) owing to its relatively short atmospheric lifespan (Zhang
```
```
et al. 2022). Furthermore, human actions, especially the combustion of
```
fossil fuels, are a primary contributor to elevated NO2 levels in the
```
environment (Finch et al. 2022). Local emission patterns generally don’t
```
change dramatically within a time span of several days. This is also
evidenced by our results from Fig. 6, which illustrates that the spatial
patterns of NO2 concentrations within the study domain remain rela-
tively consistent throughout the entire 5-day forecast period. GEOS-CF
NO, NO2, and NOy are also significant predictors, constantly ranking
among the top four in terms of importance for all five forecast days. This
is expected because, despite being biased, GEOS-CF NO2 exhibits a
moderate correlation with ground observations. Another important
```
predictor is the 10 m eastward (U) wind, which might suggest that
```
strong east wind from the ocean brings clean air into the study region,
resulting in the reduction of NO2 pollution. Latitude and longitude also
play significant roles in the model-fitting process, further demonstrating
that local emissions are the primary driver of NO2 pollution in the study
domain.
The rolling window, defined as the number of days prior to the
forecast date from which O3 and NO2 measurements were included in
the model’s training process, was found to significantly influence the
model’s overall performance. Our evaluation of rolling periods sug-
gested that a longer rolling period tended to yield more accurate O3 and
NO2 forecasts. However, the increase in model performance due to the
extension of the rolling period was insignificant, while a longer rolling
period led to a larger number of training data included and a longer
```
computation time (Tables S1 and S2). Thus, we opt for 60 days as the
```
rolling period to balance the model performance and the computational
```
Fig. 6. (continued).
```
Z. Li et al. Environment International 195 (2025) 109249
12
```
complexity, aligning with previous research (Bi et al. 2022).
```
Our RF-based forecast model has three benefits over previous
```
research (Cheng et al. 2022; Sun et al. 2021). First, conventional fore-
```
casting methods, including CTMs and time-series forecast models, either
offer spatial distributions with limited accuracy or entirely omit spatial
details. Compared to traditional techniques, our RF-based forecast
model can produce more accurate and reliable spatial patterns of NO2
and O3 concentrations, as demonstrated by our spatial CV results. Sec-
ond, the rolling period employed in this study ensured that no future
observations were used in the model training process, avoiding
improperly inflating validation performance. When a random split of the
observations from the entire period creates separate training and test
datasets, there is a high probability that the training set will contain
some same-day observations that are actually part of the test set. This
could potentially lead to artificially improved validation performance,
as the model is being trained on data that is also part of the test set.
Third, previous studies have shown that regional CTMs, such as CMAQ
and WRF-Chem, can produce O3 and NO2 forecasts with relatively high
```
accuracy. For instance, Liu et al. (2018) assessed the performance of
```
CMAQ in Hong Kong, achieving a mean correlation coefficient of 0.63
and a mean RMSE of 21.1 ppb for O3. For NO2, the mean correlation
coefficient was 0.36 with a mean RMSE of 17.4 ppb over a 3-day fore-
```
casting period. Zhou et al. (2017) conducted a thorough evaluation of
```
WRF-Chem in eastern China. For O3, the correlation coefficients ranged
from 0.62 to 0.63, with RMSEs between 26.4 and 27.9 ppb over a 72-
hour forecasting period. Additionally, for NO2, the 48-hour forecast
performance showed a correlation coefficient of 0.48 and an RMSE of
10.5 ppb. While these results fall short of the accuracy achieved by our
model forecasts, it should be noted that direct comparisons might be
biased due to differences in the geographical scope and time frames of
each individual study. Although CMAQ and WRF-Chem have the po-
tential to produce accurate O3 and NO2 forecasts at the local scale, these
models do not offer readily available forecasting output, and their
operation is computationally intensive and typically requires high-
```
performance computing resources (Beelen et al. 2009), making routine
```
O3 and NO2 forecasting less practical in resource-limited settings. In
contrast, GEOS-CF provides global 5-day operational forecasts of O3 and
NO2 concentrations that are publicly accessible in near real-time. By
combining GEOS-CF forecasts with machine learning, our framework
delivers a computationally efficient and cost-effective forecasting solu-
tion that can be implemented on personal computers, applied across the
globe, and potentially employed for high-resolution forecasting appli-
```
cations (Bi et al. 2022).
```
The limitations of our approach are twofold. First, compared to the
original GEOS-CF model, our forecasting framework could better cap-
ture the spatial distribution of O3 levels after the occurrence of special
events like tropical cyclones. However, our forecast model showed a
generally reduced ability to forecast pollutant concentrations when a
sudden change of pollutant concentrations occurred, which is particu-
larly pronounced for O3 likely due to greater fluctuations in O3 levels. It
could be attributed to factors outside the study domain, including the
transport of intense ozone pollution from adjacent provinces into the
study area and the reduction of pollution linked to strong east and south
```
winds from the ocean. In addition, Wang et al. (2011) reported that the
```
dominant westerly winds and vigorous cyclonic activity during the
spring season enhance the transportation of ozone from South Asia to
China. Note that machine learning models alone can barely capture the
variations of O3 and NO2 concentrations associated with sudden events
without extra related factors included in the model training process. This
problem could be alleviated by including foreign wildfire data or
regional CTMs with a finer spatial resolution, such as CMAQ. Second,
our forecast framework does not increase the spatial resolution of the
initial GEOS-CF forecast product. Although GEOS-CF O3 and NO2 fore-
casts can be spatially interpolated into a finer resolution, this could
introduce extra uncertainties into our O3 and NO2 forecasts and thus not
be adopted. This issue can be potentially addressed by incorporating
high-resolution remote sensing data as predictors in the forecast model
to provide extra spatial information.
5. Conclusions
For this research, we adopted an RF-based forecast model to generate
near-real-time O3 and NO2 forecasts in southeastern China with
continuous spatiotemporal coverage, spanning a 5-day period. Our
proposed forecast framework substantially improves the accuracy of the
initial GEOS-CF forecast product. In addition, our model can provide
reliable O3 and NO2 forecasts in areas without ground monitors. The
GEOS-CF ozone forecasts, the current-day convolutional layer, total
precipitation, total cloud area fraction, O3 total column density, and O3
tropospheric column density are identified as the key predictors in the
O3 forecasting model, while the current-day convolutional layer, GEOS-
```
CF NO, NO2, and NOy forecasts, 10 m Eastward (U) wind, and latitude
```
and longitude are identified as the key predictors in the NO2 forecasting
model. The forecasting model performs significantly better for NO2 than
for O3, because NO2 is less affected by emission sources outside the study
domain. Our forecasting model demonstrates improved capability over
the GEOS-CF model in capturing the spatial patterns of O3 levels after
the occurrence of special events like tropical cyclones. The results of this
study may serve as a foundation for environmental agencies to develop
effective mitigation methods for O3 and NO2 pollution in China.
CRediT authorship contribution statement
Zeyue Li: Writing – original draft, Visualization, Validation, Formal
analysis, Data curation, Investigation, Software. Jianzhao Bi: Writing –
review & editing, Methodology. Yang Liu: Writing – review & editing,
Methodology. Xuefei Hu: Writing – review & editing, Supervision, Re-
sources, Project administration, Funding acquisition, Conceptualization.
Declaration of competing interest
The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to influence
the work reported in this paper.
Appendix A. Supplementary material
Supplementary data to this article can be found online at https://doi.
org/10.1016/j.envint.2024.109249.
Data availability
Data will be made available on request.
REFERENCES
Ainsworth, E.A., Yendrek, C.R., Sitch, S., Collins, W.J., Emberson, L.D., 2012. The effectsof tropospheric ozone on net primary productivity and implications for climate
change. Annu. Rev. Plant Biol. 63, 637–661.Al Yammahi, A., Aung, Z., 2023. Forecasting the concentration of NO2 using statistical
and machine learning methods: a case study in the UAE. Heliyon 9, e12584.AlOmar, M.K., Hameed, M.M., AlSaadi, M.A., 2020. Multi hours ahead prediction of
surface ozone gas concentration: Robust artificial intelligence approach. Atmos.Pollut. Res. 11, 1572–1587.
Anenberg, S.C., Mohegh, A., Goldberg, D.L., Kerr, G.H., Brauer, M., Burkart, K.,Hystad, P., Larkin, A., Wozniak, S., Lamsal, L., 2022. Long-term trends in urban NO2
concentrations and associated paediatric asthma incidence: estimates from globaldatasets. The Lancet Planetary Health 6, e49–e58.
Bates, K.H., Jacob, D.J., 2019. A new model mechanism for atmospheric oxidation ofisoprene: global effects on oxidants, nitrogen oxides, organic products, and
secondary organic aerosol. Atmos. Chem. Phys. 19, 9613–9640.Beelen, R., Hoek, G., Pebesma, E., Vienneau, D., de Hoogh, K., Briggs, D.J., 2009.
Mapping of background air pollution at a fine spatial scale across the EuropeanUnion. Sci. Total Environ. 407, 1852–1867.
Bi, J., Knowland, K.E., Keller, C.A., Liu, Y., 2022. Combining machine learning andnumerical simulation for high-resolution PM2.5 concentration forecast. Environ. Sci.
Tech. 56, 1544–1556.
Z. Li et al. Environment International 195 (2025) 109249
13
Cakmak, S., Hebbern, C., Vanos, J., Crouse, D.L., Burnett, R., 2016. Ozone exposure andcardiovascular-related mortality in the Canadian Census Health and Environment
```
Cohort (CANCHEC) by spatial synoptic classification zone. Environ. Pollut. 214,589–599.
```
Cao, B., Yin, Z., 2020. Future atmospheric circulations benefit ozone pollution control inBeijing-Tianjin-Hebei with global warming. Sci. Total Environ. 743, 140645.
Casciaro, G., Cavaiola, M., Mazzino, A., 2022. Calibrating the CAMS European multi-model air quality forecasts for regional air pollution monitoring. Atmos. Environ.
287, 119259.Chai, T., Kim, H.-C., Pan, L., Lee, P., Tong, D., 2017. Impact of Moderate Resolution
Imaging Spectroradiometer Aerosol Optical Depth and AirNow PM2.5 assimilationon Community Multi-scale Air Quality aerosol predictions over the contiguous
United States. J. Geophys. Res. Atmos. 122, 5399–5415.Cheng, M., Fang, F., Navon, I.M., Zheng, J., Tang, X., Zhu, J., Pain, C., 2022. Spatio-
Temporal Hourly and Daily Ozone Forecasting in China Using a Hybrid MachineLearning Model: Autoencoder and Generative Adversarial Networks. J. Adv. Model.
Earth Syst. 14, e2021MS002806.Chu, Y., Liu, Y., Li, X., Liu, Z., Lu, H., Lu, Y., Mao, Z., Chen, X., Li, N., Ren, M., Liu, F.,
Tian, L., Zhu, Z., Xiang, H., 2016. A Review on Predicting Ground PM2.5Concentration Using Satellite Aerosol Optical Depth. Atmos. 7, 129.
Eslami, E., Choi, Y., Lops, Y., Sayeed, A., 2020. A real-time hourly ozone predictionsystem using deep convolutional neural network. Neural Comput. & Applic. 32,
8783–8797.Feng, R., Zheng, H.-J., Zhang, A.-R., Huang, C., Gao, H., Ma, Y.-C., 2019. Unveiling
tropospheric ozone by the traditional atmospheric model and machine learning, andtheir comparison:A case study in hangzhou, China. Environ. Pollut. 252, 366–378.
Finch, D.P., Palmer, P.I., Zhang, T., 2022. Automated detection of atmospheric NO2plumes from satellite data: a tool to help infer anthropogenic combustion emissions.
Atmos. Meas. Tech. 15, 721–733.Hoffmann, B., Luttmann-Gibson, H., Cohen, A., Zanobetti, A., Souza, C.D., Foley, C.,
Suh, H.H., Coull, B.A., Schwartz, J., Mittleman, M., Stone, P., Horton, E., Gold, D.R.,2012. Opposing effects of particle pollution, ozone, and ambient temperature on
arterial blood pressure. Environ. Health Perspect. 120, 241–246.Hou, T., Yu, S., Jiang, Y., Chen, X., Zhang, Y., Li, M., Li, Z., Song, Z., Li, P., Chen, J.,
Zhang, X., 2022. Impacts of chemical initial conditions in the WRF-CMAQ model onthe ozone forecasts in Eastern China. Aerosol Air Qual. Res. 22, 210402.
Hu, X., Waller, L.A., Lyapustin, A., Wang, Y., Al-Hamdan, M.Z., Crosson, W.L., Estes, M.G., Estes, S.M., Quattrochi, D.A., Puttaswamy, S.J., Liu, Y., 2014. Estimating ground-
level PM2.5 concentrations in the Southeastern United States using MAIAC AODretrievals and a two-stage model. Remote Sens. Environ. 140, 220–232.
Hu, X., Belle, J.H., Meng, X., Wildani, A., Waller, L.A., Strickland, M.J., Liu, Y., 2017.Estimating PM2.5 concentrations in the conterminous United States using the
random forest approach. Environ. Sci. Tech. 51, 6936–6944.Jerrett, M., Burnett, R.T., Pope, C.A., Ito, K., Thurston, G., Krewski, D., Shi, Y., Calle, E.,
Thun, M., 2009. Long-Term Ozone Exposure and Mortality. N. Engl. J. Med. 360,1085–1095.
Keller, C.A., Knowland, K.E., Duncan, B.N., Liu, J., Anderson, D.C., Das, S., Lucchesi, R.A., Lundgren, E.W., Nicely, J.M., Nielsen, E., Ott, L.E., Saunders, E., Strode, S.A.,
Wales, P.A., Jacob, D.J., Pawson, S., 2021. Description of the NASA GEOSComposition Forecast Modeling System GEOS-CF v1.0. J. Adv. Model. Earth Syst. 13,
e2020MS002413.Khomenko, S., Cirach, M., Pereira-Barboza, E., Mueller, N., Barrera-G´omez, J., Rojas-
Rueda, D., de Hoogh, K., Hoek, G., Nieuwenhuijsen, M., 2021. Premature mortalitydue to air pollution in European cities: a health impact assessment. The Lancet
Planetary Health 5, e121–e134.Knowland, K.E., Keller, C.A., Wales, P.A., Wargan, K., Coy, L., Johnson, M.S., Liu, J.,
Lucchesi, R.A., Eastham, S.D., Fleming, E., Liang, Q., Leblanc, T., Livesey, N.J.,Walker, K.A., Ott, L.E., Pawson, S., 2022. NASA GEOS Composition Forecast
Modeling System GEOS-CF v1.0: Stratospheric Composition. J. Adv. Model. EarthSyst. 14, e2021MS002852.
Liu, T., Lau, A.K.H., Sandbrink, K., Fung, J.C.H., 2018. Time Series Forecasting of AirQuality Based On Regional Numerical Modeling in Hong Kong. J. Geophys. Res.
Atmos. 123, 4175–4196.Liu, W., Li, X., Chen, Z., Zeng, G., Le´on, T., Liang, J., Huang, G., Gao, Z., Jiao, S., He, X.,
Lai, M., 2015. Land use regression models coupled with meteorology to modelspatial and temporal variability of NO2 and PM10 in Changsha, China. Atmos.
Environ. 116, 272–280.Lobell, D.B., Di Tommaso, S., Burney, J.A., 2022. Globally ubiquitous negative effects of
nitrogen dioxide on crop growth. Sci. Adv. 8, eabm9909.Ma, R., Ban, J., Wang, Q., Zhang, Y., Yang, Y., He, M.Z., Li, S., Shi, W., Li, T., 2021.
Random forest model based fine scale spatiotemporal O3 trends in the Beijing-Tianjin-Hebei region in China, 2010 to 2017. Environ. Pollut. 276, 116635.
```
Martínez-Lazcano, J.C., Gonz´alez-Guevara, E., Rubio, M.d.C., Franco-P´erez, J., Custodio,V., Hern´andez-Cer´on, M., Livera, C., & Paz, C. (2013). The effects of ozone exposure
```
and associated injury mechanisms on the central nervous system. Reviews in theNeurosciences, 24, 337–352.
Mendes, L., Monjardino, J., Ferreira, F., 2022. Air Quality Forecast by StatisticalMethods: Application to Portugal and Macao. Front. Big Data 5.
Meng, X., Wang, W., Shi, S., Zhu, S., Wang, P., Chen, R., Xiao, Q., Xue, T., Geng, G.,Zhang, Q., Kan, H., Zhang, H., 2022. Evaluating the spatiotemporal ozone
characteristics with high-resolution predictions in mainland China, 2013–2019.Environ. Pollut. 299, 118865.
Mo, Y., Li, Q., Karimian, H., Fang, S., Tang, B., Chen, G., Sachdeva, S., 2020. A novelframework for daily forecasting of ozone mass concentrations based on cycle
```
reservoir with regular jumps neural networks. Atmos. Environ. 220, 117072.Rai, R., & Agrawal, M. (2012). Impact of Tropospheric Ozone on Crop Plants.
```
Proceedings of the National Academy of Sciences, India Section B: BiologicalSciences, 82, 241-257.
Sayeed, A., Choi, Y., Eslami, E., Jung, J., Lops, Y., Salman, A.K., Lee, J.-B., Park, H.-J.,Choi, M.-H., 2021. A novel CMAQ-CNN hybrid model to forecast hourly surface-
ozone concentrations 14 days in advance. Sci. Rep. 11, 10891.Song, J., Wang, Y., Zhang, Q., Qin, W., Pan, R., Yi, W., Xu, Z., Cheng, J., Su, H., 2023.
Premature mortality attributable to NO2 exposure in cities and the role of builtenvironment: A global analysis. Sci. Total Environ. 866, 161395.
Sun, H., Fung, J.C.H., Chen, Y., Chen, W., Li, Z., Huang, Y., Lin, C., Hu, M., Lu, X., 2021.Improvement of PM2.5 and O3 forecasting by integration of 3D numerical
simulation with deep learning techniques. Sustain. Cities Soc. 75, 103372.Tang, X., Zhu, J., Wang, Z.F., Gbaguidi, A., 2011. Improvement of ozone forecast over
Beijing based on ensemble Kalman filter with simultaneous adjustment of initialconditions and emissions. Atmos. Chem. Phys. 11, 12901–12916.
Tang, X., Gao, X., Li, C., Zhou, Q., Ren, C., Feng, Z., 2020. Study on spatiotemporaldistribution of airborne ozone pollution in subtropical region considering
socioeconomic driving impacts: A case study in Guangzhou China. Sustain. CitiesSoc. 54, 101989.
Tang, Y., Pagowski, M., Chai, T., Pan, L., Lee, P., Baker, B., Kumar, R., Delle Monache, L.,Tong, D., Kim, H.C., 2017. A case study of aerosol data assimilation with the
Community Multi-scale Air Quality Model over the contiguous United States using3D-Var and optimal interpolation methods. Geosci. Model Dev. 10, 4743–4758.
```
Travis, K.R., Jacob, D.J., 2019. Systematic bias in evaluating chemical transport modelswith maximum daily 8&thinsp;h average (MDA8) surface ozone for air quality
```
```
applications: a case study with GEOS-Chem v9.02. Geosci. Model Dev. 12,3641–3648.
```
Vasseur, S.P., Aznarte, J.L., 2021. Comparing quantile regression methods forprobabilistic forecasting of NO2 pollution levels. Sci. Rep. 11, 11592.
Wang, Y., Gao, W., Wang, S., Song, T., Gong, Z., Ji, D., Wang, L., Liu, Z., Tang, G.,Huo, Y., Tian, S., Li, J., Li, M., Yang, Y., Chu, B., Pet¨aj¨a, T., Kerminen, V.-M., He, H.,
Hao, J., Kulmala, M., Wang, Y., Zhang, Y., 2020. Contrasting trends of PM2.5 andsurface-ozone concentrations in China from 2013 to 2017. Natl. Sci. Rev. 7,
1331–1339.Wang, K., Yuan, Y., Wang, Q., Yang, Z., Zhan, Y., Wang, Y., Wang, F., Zhang, Y., 2023.
Incident risk and burden of cardiovascular diseases attributable to long-term NO2exposure in Chinese adults. Environ. Int. 178, 108060.
Wang, Y., Zhang, Y., Hao, J., Luo, M., 2011. Seasonal and spatial variability of surfaceozone over China: contributions from background and domestic pollution. Atmos.
Chem. Phys. 11, 3511–3525.Wen, W., Shen, S., Liu, L., Ma, X., Wei, Y., Wang, J., Xing, Y., Su, W., 2021. Comparative
```
Analysis of PM2.5 and O3 Source in Beijing Using a Chemical Transport Model.Remote Sens. (Basel) 13, 3457.
```
Xu, X., Zhang, T., 2020. Spatial-temporal variability of PM2.5 air quality in Beijing,China during 2013–2018. J. Environ. Manage. 262, 110263.
Yendrek, C.R., Leisner, C.P., Ainsworth, E.A., 2013. Chronic ozone exacerbates thereduction in photosynthesis and acceleration of senescence caused by limited N
availability in Nicotiana sylvestris. Glob. Chang. Biol. 19, 3155–3166.Yuval, Broday, D.M., Alpert, P., 2012. Exploring the applicability of future air quality
predictions based on synoptic system forecasts. Environ. Pollut. 166, 65–74.Zhan, Y., Luo, Y., Deng, X., Grieneisen, M.L., Zhang, M., Di, B., 2018. Spatiotemporal
prediction of daily ambient ozone levels across China using random forest for humanexposure assessment. Environ. Pollut. 233, 464–473.
```
Zhang, Z., Liu, Y., Liu, H., Hao, A., Zhang, Z., 2022. The impact of lockdown on nitrogendioxide (NO2) over Central Asian countries during the COVID-19 pandemic.
```
Environ. Sci. Pollut. Res. 29, 18923–18931.Zhang, H., Wang, J., García, L.C., Ge, C., Plessel, T., Szykman, J., Murphy, B., Spero, T.L.,
2020. Improving surface PM2.5 forecasts in the United States using an ensemble ofchemical transport model outputs: 1. bias correction with surface observations in
nonrural areas. J. Geophys. Res. Atmos. 125, e2019JD032293.Zhao, D., Xu, H., Li, Y., Yu, Y., Duan, Y., Xu, X., Chen, L., 2024. Locally opposite
responses of the 2023 Beijing–Tianjin–Hebei extreme rainfall event to globalanthropogenic warming. npj Climate Atmosp. Sci. 7, 38.
Zheng, H., Cai, S., Wang, S., Zhao, B., Chang, X., Hao, J., 2019. Development of a unit-based industrial emission inventory in the Beijing–Tianjin–Hebei region and
resulting improvement in air quality modeling. Atmos. Chem. Phys. 19, 3447–3462.Zhou, G., Xu, J., Xie, Y., Chang, L., Gao, W., Gu, Y., Zhou, J., 2017. Numerical air quality
forecasting over eastern China: An operational application of WRF-Chem. Atmos.Environ. 153, 94–108.
Z. Li et al. Environment International 195 (2025) 109249
14