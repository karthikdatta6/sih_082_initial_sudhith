

_Article_ 

# **Machine Learning-Based Ground-Level NO2 Estimation in Istanbul: A Comparative Analysis of Sentinel-5P and GEOS-CF** 

### **Nur Yagmur Aydin** 

Geomatics Engineering Department, Engineering Faculty, Gebze Technical University, Kocaeli 41400, Türkiye; nyagmur@gtu.edu.tr 

### **Abstract** 

Nitrogen dioxide (NO2) poses severe risks to human health and the environment, especially in densely populated megacities. Ground-based air quality monitoring stations provide hightemporal-resolution data but are spatially limited, while satellite observations offer broad coverage but measure column densities rather than surface concentrations. To overcome these limitations, this study integrates ground-based observations with satellite-derived NO2 from Sentinel-5P TROPOMI and GEOS-CF products to estimate ground-level NO2 in Istanbul using machine learning (ML) approaches. Three ML algorithms (RF, XGB, and CB) were tested on two datasets spanning 2019–2024 at ~1 km resolution, incorporating 20 features, including topographic, meteorological, environmental, and demographic variables. Among models, CB achieved the best performance (R: 0.686, RMSE: 16.23 µg/m<sup>3</sup> , and MAE: 11.75 µg/m<sup>3</sup> in the test dataset) with the Sentinel-5P dataset, successfully capturing spatial and seasonal variations in ground-level NO2 both quantitatively and qualitatively. SHAP analysis revealed that regarding satellite-derived NO2, anthropogenic indicators such as population density, road length, and digital elevation model were the most influential features, while meteorological factors contributed secondarily. Despite the lower spatial resolution of GEOS-CF data, both Sentinel-5P and GEOS-CF datasets supported reliable model outputs. This study provides the first ML-based ground-level NO2 estimation framework for the Istanbul Metropolitan City. 

**Keywords:** Sentinel-5P; Geos-CF; ground-level NO2; Istanbul; CatBoost; SHAP; machine learning 

Academic Editor: Hyo Choi 

Received: 22 September 2025 Revised: 10 October 2025 Accepted: 13 October 2025 Published: 13 October 2025 

**Citation:** Yagmur Aydin, N. Machine Learning-Based Ground-Level NO2 Estimation in Istanbul: A Comparative Analysis of Sentinel-5P and GEOS-CF. _Appl. Sci._ **2025** , _15_ , 10997. https:// doi.org/10.3390/app152010997 

**Copyright:** © 2025 by the author. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/). 

## **1. Introduction** 

As one of the main air pollutants, nitrogen dioxide (NO2) is produced by natural and anthropogenic sources, including fossil fuel burning from transportation, industrial activities, and power plants [1]. Exposure to NO2 is associated with a range of health issues, such as cardiovascular and respiratory diseases, lung cancer, and premature death [2–4]. Beyond its health impacts, NO2 also adversely affects the natural environment. It contributes to the formation of tropospheric ozone and aerosol nitrates, leading to acid rain and reduced visibility [5]. Moreover, high concentrations of NO2 can damage crops and vegetation by reducing yields and inhibiting plant growth [6]. 

The rapid growth of megacity populations, driven by economic and technological development, has further increased NO2 emissions from human activities. Several studies have demonstrated the strong connection between economic development and groundlevel NO2 concentrations [7,8]. Over the past three decades, megacities have contributed significantly to the rise in anthropogenic NO2 emissions, making them critical areas for air pollution mitigation and control [9,10]. Therefore, continuous monitoring and accurate 

_Appl. Sci._ **2025** , _15_ , 10997 

https://doi.org/10.3390/app152010997 

_Appl. Sci._ **2025** , _15_ , 10997 

2 of 19 

estimation of ground-level NO2 concentrations in megacities are of vital importance for tracking air pollution and developing effective action plans [11]. 

Air quality monitoring stations are sustainable systems established in regions where air pollution monitoring is essential. These stations provide high spatial and temporal resolution data, typically with hourly measurements. However, as point-based systems, their spatial coverage is limited. Furthermore, since the monitoring of anthropogenic emissions is often prioritized, the distribution of stations is concentrated in urban areas, resulting in a lack of spatial balance [12,13]. On the contrary, remote sensing technologies enable spatial and temporal analyses through their wide coverage and repeated data acquisition [14]. Despite these advantages, remote sensing has limitations, such as the inability to collect data under cloudy conditions and its measurement of column density rather than surface or near-surface concentrations [13,15]. Nevertheless, numerous studies have demonstrated that satellite-observed tropospheric NO2 column density correlates with ground-level NO2 due to its short atmospheric lifetime and its formation from anthropogenic activities [1,16]. Considering the respective advantages and limitations of satellite and ground-based monitoring systems, their combined use provides complementary insights, supporting the development of more comprehensive and accurate models. 

Several approaches have been employed to estimate ground-level NO2, including traditional statistical methods such as kriging, land use regression, and geographically and temporally weighted regression [16]. However, machine learning (ML) models have shown superior performance by effectively capturing complex nonlinear relationships, accounting for interactions among diverse variables, and achieving more accurate predictions [13,17,18]. ML-based studies have been applied across various scales, including local [19–21], regional [22,23], and national [10,17,24–26] levels. In addition, explainable artificial intelligence (XAI) techniques, such as Shapley Additive exPlanations (SHAP), have been increasingly used to identify and interpret the environmental, meteorological, and topographic drivers influencing ground-level NO2 estimates [22,27,28]. 

Within the scope of this study, ground-level NO2 concentrations in Istanbul were estimated using ML algorithms by integrating ground-based air quality monitoring data with satellite-derived tropospheric column of NO2, including Sentinel-5P TROPOMI and Geos-CF datasets. Three machine learning algorithms were applied to a dataset spanning 2019–2024 at a spatial resolution of ~1 km, with a focus on seasonal variations. Both qualitative and quantitative evaluations were conducted. To improve prediction accuracy, a comprehensive set of topographic, meteorological, environmental, and demographic variables was incorporated. Model interpretability was ensured using SHAP, a widely adopted explainable AI method [29], to identify the most influential features of NO2 variability. The study will address the following research questions: 

- How accurately can ground-level NO2 in metropolitan cities be estimated using satellite-derived tropospheric NO2 data? 

- How do ground-level NO2 estimates differ when using NO2 data with varying spatial resolutions? 

- To what extent do environmental and anthropogenic factors influence the prediction of ground-level NO2? 

In this context, this study represents the first ML-based ground-level NO2 estimation framework for Istanbul, Türkiye’s most populous and touristic city, providing critical insights for air quality management and policymaking. 

_Appl. Sci._ **2025** , _15_ , 10997 

3 of 19 

## **2. Study Area and Datasets** 

### _2.1. Study Area and Ambient Air Quality Monitoring Station Data_ 

Istanbul is Türkiye’s most populous metropolitan city, with a population of approximately 16 million. Its role as a bridge connecting Asia and Europe has made it a major transportation, industrial, and tourism hub (Figure 1). Due to internal migration and increasing population, residential areas have increased significantly in the last 30 years to meet the city’s housing needs [30,31], and it is estimated that they will increase further [32]. Therefore, monitoring air quality in cities with intense human activity is gaining importance. To this end, the Istanbul Metropolitan Municipality and the Ministry of Environment, Urbanization, and Climate Change have established air quality monitoring stations in the city. The distribution of air quality monitoring stations is shown in Figure 1. 



<!-- Start of picture text -->
o4 2 mee NO<br>Y pt<br>isa cases<br>One ay a,<br>PeSo om ee4<br>x IX<br>8<br>=<br>x<br>2<br>=<br>+ Air Quality Monitoring Stations<br><!-- End of picture text -->



<!-- Start of picture text -->
(Gi Istanbul Qo BD 291i<br>(7) National Boundary =_<br>28°0'E 28°30'E 29°0'E 29°30'E<br><!-- End of picture text -->

**Figure 1.** The location of Türkiye and Istanbul, along with the distribution of air quality monitoring stations. 

There are 40 air quality stations in Istanbul. However, only 32 of these stations measure the NO2 parameter. The distribution of stations is concentrated around the Bosphorus, where transportation, touristic and industrial activities, and residential areas are concentrated. This allows monitoring only in areas where threats are identified, but not in areas without air quality stations. In air quality monitoring stations, parameters are measured hourly and serviced online (https://havakalitesi.ibb.gov.tr/, accessed on 1 August 2025). 

Hourly collected station data were filtered based on the satellite passing time (between 1.00 and 2.00 p.m. in UTC+3 time) between the years of 2019 and 2024, and the data from both hours each day were averaged. To minimize the effects of abnormal values on the model, values greater than 300 and less than one were eliminated from the data, the same as with Chi et al. (2022) [17]. 

_Appl. Sci._ **2025** , _15_ , 10997 

4 of 19 

### _2.2. Datasets_ 

In the study, various datasets were identified through a literature review and collected from different sources, taking into account the matching of time intervals, as listed in Table 1. 

**Table 1.** List of input and output features used in the study. 

|**Data Type**|**Name**|**Variable**|**Data Source**|**Input/Output**|
|---|---|---|---|---|
|Ground Monitoring|Ground-based NO2<br>Station Data<br>(Hourly average)|NO2 measurements|https://havakalitesi.<br>ibb.gov.tr/, accessed on<br>1 August 2025|Output|
|Satellite Air Quality<br>Product|Sentinel-5P TROPOMI<br>(Daily)|Tropospheric vertical column<br>of NO2|https://earthengine.<br>google.com,<br>accessed on<br>1 August 2025<br>|Input|
||Geos-CF<br>(Hourly average)|Hourly average Nitrogen dioxide<br>(NO2, MW = 46.00 g mol<sup>_−_1</sup>)<br>tropospheric column density|https://earthengine.<br>google.com,<br>accessed on<br>1 August 2025|Input|
|||Dust optical depth at 550 nm<br>(AOD550_Dust)||Input|
|||Surface geopotential height (PHIS)<br>Surface pressure (PS)||Input<br>Input|
|||<br>Specific humidity (Q)||Input|
|||Relative humidity after moist (RH)|https://earthengine.|Input|
|Climate|Geos-CF<br>(Hourly average)|Sea level pressure (SLP)<br>2-m air temperature (T2M)<br>Total precipitation (TPREC)<br>Surface skin temperature (TS)|google.com,<br>accessed on<br>1 August 2025|Input<br>Input<br>Input<br>Input|
|||10-m eastward wind (U10M)||Input|
|||<br>10-m northward wind (V10M)<br>Mid-layer heights (ZL)<br>||Input<br>Input|
|||Planetary boundary layer<br>height (ZPBL)||Input|
||MODIS<br>(Daily)|Normalized Difference Vegetation<br>Index (NDVI)|https://earthengine.<br>google.com,<br>accessed on<br>1 August 2025<br>https://earthengine.|Input|
|Society|VIIRS<br>(Daily)|Nighttime light (NTL)|google.com,<br>accessed on<br>1 August 2025<br>https:|Input|
||OpenStreetMap|Road Length (RL)|//www.geofabrik.de,<br>accessed on<br>1 August 2025|Input|
||TUIK<br>(Annual)|Population Density (PD)|<br>https://biruni.tuik.gov.<br>tr/medas/, accessed on<br>1 August 2025|Input|
||||https://earthengine.<br>||
|Topography|SRTM|Digital Elevation Model (DEM)|google.com,<br>accessed on|Input|
||||1 August 2025||
|-|-|Day of Year (DOY)|-|Input|



2.2.1. Sentinel-5P TROPOMI Tropospheric NO2 Columns 

The Sentinel-5P satellite was launched on 13 October 2017, and was designed for monitoring the atmosphere and air pollution, ozone-layer monitoring, climate change and aviation safety at high spatial resolution under the Copernicus mission. The TROPOMI 

_Appl. Sci._ **2025** , _15_ , 10997 

5 of 19 

(TROPOspheric Monitoring Instrument) sensor collects solar radiation backscattered from the Earth and atmosphere. Its products are served in two processing levels with a spatial sampling of approximately 3.5 _×_ 5.5 km since 6 August 2019. However, the Level-2 data is stored as Level-3 OFL with a spatial sampling of approximately 1 _×_ 1 km in Google Earth Engine (GEE) [33]. The Sentinel-5P TROPOMI dataset was extracted using the GEE cloud computing platform, which stores archive and up-to-date datasets and enables data processing and geospatial analysis. 

### 2.2.2. Satellite-Based Variables 

Various topographical, environmental, and meteorological factors were included in the model to enhance the data space and improve model accuracy, as well as their influence on NO2. The Shuttle Radar Topography Mission (SRTM) digital elevation model (DEM) [34] was used as a topographical factor. SRTM is a global DEM dataset with a 30 m spatial resolution. Normalized Difference Vegetation Index (NDVI) [35] was used as the environmental factor to identify green and non-green areas. NDVI data were calculated from MODIS Nadir Bidirectional Reflectance Distribution Function Adjusted Reflectance (NBAR) data, which has a spatial resolution of 500 m and has been providing data since 2000 [36]. Nighttime light (NTL) data were used to identify urban areas and non-green areas, representing the artificial lights of the settlements and human activities [37]. Considering the impact of anthropogenic activities on NO2 formation, NTL data spatially well characterize the areas where human activities and hence NO2 emissions are concentrated [38]. For this purpose, NTL data were obtained with the Visible Infrared Imaging Radiometer Suite (VIIRS) with a spatial resolution of 500 m [39]. 

The climate variables were provided by the Goddard Earth Observing System Composition Forecast (Geos-CF). Geos-CF, developed by the National Aeronautics and Space Administration (NASA), includes globally produced three-dimensional distributions of atmospheric composition with a spatial resolution of approximately 27 km. Geos-CF products cover atmospheric replay to time-average one-hour data by combining meteorological, atmospheric, and chemical collections [40]. Fourteen bands of the Geos-CF were used in the study, as given in Table 1. To match the Geos-CF dataset with Sentinel-5P, the GeosCF dataset was temporally filtered based on the satellite passing time of the Sentinel-5P between the years of 2019 and 2024, and the Geos-CF data taken in both hours (between 1.00 and 2.00 p.m. in UTC+3 time) for each day were averaged to provide temporal consistency of the datasets. 

### 2.2.3. Auxiliary Variables 

To consider the effects of social factors, population density (PD) and road length (RL) variables were included in the estimation model. Population data were provided from the Turkish Statistical Institute (TUIK), and population density was computed by dividing the population values by the area of each district. Road data was obtained from the road layer shared by OpenStreetMap. Within the scope of the study, 0.1 _×_ 0.1<sup>_◦_</sup> grid network was created for the study area, and the road lengths within each grid were calculated. In addition to meteorological, environmental, and social factors, the day of the year (DOY) was included in the model. These auxiliary data were used to detect the density of human activities both temporally and spatially [41]. 

## **3. Methodology** 

The study was conducted in three parts: data preprocessing and feature extraction, model development with ML algorithms and model evaluations, as shown in Figure 2. 

_Appl. Sci._ **2025** , _15_ , 10997 

6 of 19 



<!-- Start of picture text -->
DATABASE =a<br>~.-Data Partition $n<br><!-- End of picture text -->



<!-- Start of picture text -->
Evaluations 7<br>Pete tr tc ee eeeee<br>'i<br>i i<br>lee See wre ects Ps desea cata ee cata eet ee sete seen ere Perera eres areas<br><!-- End of picture text -->

**Figure 2.** Workflow of the study. 

### _3.1. Data Preprocessing and Feature Extraction_ 

Satellite images data, including Sentinel-5P, Geos-CF, MODIS NDVI, SRTM DEM, and VIIRS NTL, were extracted with the locations of air quality monitoring stations using GEE, and data were matched both temporally and spatially. Data were divided into three parts: training, validation, and test. While the training data covers the years 2019–2022, the validation data covers the year 2023, and the test data covers the year 2024. 

Two different data groups were generated to measure the effect of NO2 data sources. With this purpose, the tropospheric column density of NO2 gathered from Sentinel-5P was used as input in the first data group, while the NO2 tropospheric column density data from Geos-CF were used as input in the second data group, along with 19 other variables. 

In the phase of applying the created models to all images, 0.1 _×_ 0.1<sup>_◦_</sup> grid network was created, data was collected on the GEE platform, and the resulting thematic maps were produced with the best-performing model. 

### _3.2. Model Development with Machine Learning_ 

In order to estimate surface NO2 concentrations, the best appropriate model must generate the most accurate results. Therefore, three different machine learning algorithms, namely Random Forest (RF) Regression, Extreme Gradient Boosting (XGBoost) Regression (XGB), and CatBoost Regression (CB), were chosen based on their success in similar studies [13,19,26,42]. Within the scope of the study, Optuna [43] was used in the hyperparameter optimization of algorithms. 

_Appl. Sci._ **2025** , _15_ , 10997 

7 of 19 

RF, introduced by Breiman (2001) [44], is one of the most prominent machine learning algorithms for air quality assessment research owing to its robust predictive performance and efficiency [45,46]. RF divides the original training dataset into random subsets and constructs an ensemble of decision trees. The training process is performed utilizing 2/3 of these subsets, while the remaining subsets are responsible for evaluating the model’s accuracy [47]. The majority voting approach is utilized to identify the final label of samples. 

XGB was presented by Chen and Guestrin (2016) [48] as an advanced tree-based machine learning algorithm based on boosting theories. The main principle of XGB is the sequential refinement of weak learners within an ensemble [49]. The training process begins with a base model by allocating equal weights to all samples, and labels are predicted. In subsequent iterations, incorrectly estimated samples are assigned higher weights to fix their labels [50,51]. Unlike other ML models, XGB incorporates regularization techniques and optimized loss functions to reduce overfitting and enhance generalization [52]. 

CB, one of the latest members of tree-based algorithms, was developed by Yandex to effectively handle the challenges associated with categorical features [53]. Automatic processing of categorical data and missing values through ordered boosting eliminates manual preprocessing requirements [54]. Additionally, CB constructs a symmetric tree structure to mitigate overfitting and enables GPU acceleration for large-scale datasets, resulting in fast and superior predictive performance [55,56]. 

Moreover, in this study, SHAP, an XAI framework based on game theory, was employed to interpret model behavior by quantifying the contribution of each feature to predictions in terms of both magnitude and direction [29]. SHAP values represent the marginal effect of a feature compared to the dataset’s average prediction, while aggregated absolute SHAP values provide global feature importance [57]. This approach enables both local and global interpretation, facilitating the assessment of feature relevance, the validation of model reliability against domain knowledge, and highlighting the role of satellite observations in modeling ground-level NO2 concentrations. 

### _3.3. Model Evaluation and Accuracy Assessment_ 

To assess model accuracy, three accuracy metrics were used within the study: Pearson correlation coefficient (R), Root Mean Square Error (RMSE), and Mean Absolute Error (MAE). The equations of metrics are, respectively, given as follows: 







where _xi_ is the measured value, _x_ ˆ _i_ is the model-estimated NO2 value, n is the total number of samples in the validation dataset, and _xm_ and _x_ ˆ _m_ represent the means of the measured and estimated values, respectively. 

## **4. Results** 

### _4.1. Correlation Analysis Between Variables_ 

Firstly, the correlation coefficients between the features are calculated using the Pearson Correlation coefficient and are given in Figure 3. The results indicate that the correlation coefficients of S5P and Geos-CF NO2 data with ground station data (Sta_NO2) were de- 

_Appl. Sci._ **2025** , _15_ , 10997 

8 of 19 

termined to be close to each other, 0.38 and 0.31, respectively. However, the correlation coefficient between S5P and Geos-CF NO2 data was determined to be 0.68. NDVI has a negative correlation with road length (RL) and population density (PD), with correlation coefficients of _−_ 0.66 and _−_ 0.57, respectively. The correlation coefficient between PD and RL was 0.59. When the correlations between Geos-CF bands were examined, the highest correlations were determined as 2-m air temperature (T2M)-surface skin temperature (TS) 0.97, sea level pressure (SLP)-surface pressure (PS) 0.90, mid layer heights (ZL)-surface geopotential height (PHIS) 0.86, eastward wind (U10M)-northward wind (V10M) 0.86, T2M-specific humidity (Q) 0.81, TS-Q 0.75, respectively. 



<!-- Start of picture text -->
Correlation Heatmap of Variables<br>s5P_Noz 0.07 012 0.13 0.03 0.03 -0.03 [0.23/E0134) 0.03 [l026/50131) -0.02 0.26)021 9030) 0.03 LSE 016 007 00<br>‘Sta_NO2 100 eed O31) 0.02 011 0.03 003 0.14 0.06 005 0.09 0.02 0.10 015 021 001 0.18 031, 0.26 §Oa7)<br>DEM "0.07 -0.22 MUN 0.23 0.38 0.00 001 017 0.08 000 001 001 003 0.00 006 000 000 019 004 001 016 019 07s<br>RL - 0.12 0.23 PEFR 0.01 001 015 0.10 0.06 016 0.02 005 0.00 002 000 001 018 009 015<br>AD DUSTDOY-0030.03 0 .0211 0 .0001 0.000 1 0 .0100 015 0.15(§MIM 010 0 .0211 0 53)05 0 .00 1 0. 01 2 (034)005 0.01 (030002 01 27 0 .0815 00 31 1927O11 0 .0609 00 51 0 .0201 050<br>PHIS 0.03 0.03 017 015 0.09 010 0.10 RBMWS9) 002 0.11 0.02 011 0.01 004 000 0.01 BRR 014 0.01 013 009<br>PS 1023) 003 0.08 0.10 003 0.02 011 5039) 021 (GIES) 0.02 OMB] 0.14 0.07 HOMBL-0.18 018 005 0.05 -025<br>LPRH-003Q-034 1026] 0.140.06005 0.00001001 0.060.160.02 0.040.010.12 0.010.02O33) 0010.12006 0.110.02002 021RIEDNPIES)007 017007(EMM EERE017 FerryBUBMPI) 0.030.10009 Bes4PETfy 0.24027012 0310.06022 0.170.05001 GOIES]0.070.15019 035021 0.02013015 0.050.09000 |aco<br>T2M-931 0.09 003 005 004 (O34) 005 011 ERS CRO ATMS 001 REM036 0039) 013 OMNI 026 002 001<br>TPREC "0.02 0.02 0.00 0.00 001 001 0.01 001-002 010 009 -0.03 001 MIRE 0.00 000 002 0.01 -0.3 0.00 0.00 0.00 |o2s<br>UloM75 1.26021 0.10015 00 60 00 20 0.03006 017(O30) 0 01 2 00 40 0.14 0.24ECTS027UE0.12 10.36RDEEN 0.000 0 /037RGN -03BERMNRTE7/041) 0 .0502 MOMMI0(041) 0 1 79 0.02002 0 .0201<br>‘V1OM -030 021 000 001 0.00 0.08 015 0.01 0.07 031 022 0.06 039 0.02 041 (RREREN 0.00 Eukrg 0.24 0.05 0.00 -0.50<br>21-003 001 019° 018 0.09 003<br>001 (OEEBWMB]001 0.17 0.05 013 0.01 005 002 000 BRM 012 000 016 011<br>GeosZPBLNOVI-0.16wn. 007 NO2 UEBIOSH) 034)0.18(037-0190.26 0.01 0.04016 009 015 32/002 016 0.09 0.06011005 0.27)001009001 0.010.13009014 0.180.05018005 359050.19013 (NMS)909007015 -0.150.02000021 0.26001002MAN 0.009000.03000 RUMMNCOMNNIEDS9020.19002 0.010.02017 0.000.05024 0.160110.12000 0.280.07006 0:28)0.21009BR WO3TIE0.210.07 37)0090.06 “100-0.75<br>s 8 E228 eee ok BEE<br>BS 2 a Bg g 8 Ree =<br>3 28 g 8 B<br>& 8 3<br><!-- End of picture text -->

**Figure 3.** The correlation heatmap of the features obtained on a daily basis. 

### _4.2. Accuracy Assessment and Seasonal Thematic Maps_ 

Three ML algorithms were tested with accuracy metrics including R, RMSE, and MAE for each data part, and the results are given in Table 2. The best-performing model was highlighted for both Sentinel-5P and Geos-CF datasets. 

**Table 2.** Accuracy assessment results of the methods during training, validation, and test steps. 

|||||**Sentine**|**l-5P**|||||
|---|---|---|---|---|---|---|---|---|---|
|**Data**|**Tr**|**ain (2019–20**|**22)**|**V**|**alidation (20**|**23)**||**Test (2024)**||
|**Model/Metric**|**R**|**RMSE**<br>**(µg/m**<sup>**3**</sup>**)**|**MAE**<br>**(µg/m**<sup>**3**</sup>**)**|**R**|**RMSE**<br>**(µg/m**<sup>**3**</sup>**)**|**MAE**<br>**(µg/m**<sup>**3**</sup>**)**|**R**|**RMSE**<br>**(µg/m**<sup>**3**</sup>**)**|**MAE**<br>**(µg/m**<sup>**3**</sup>**)**|
|RF|0.820|16.302|11.458|0.660|17.909|12.880|0.666|16.645|12.150|
|XGB|0.945|9.157|6.405|0.657|18.274|13.111|0.638|17.605|12.766|
|CB|0.827|15.772|11.121|**0.669**|**17.743**|**12.658**|**0.686**|**16.232**|**11.746**|
|||||**Geos-**|**CF**|||||
|**Model/Metric**|**R**|**RMSE**<br>**(µg/m**<sup>**3**</sup>**)**|**MAE**<br>**(µg/m**<sup>**3**</sup>**)**|**R**|**RMSE**<br>**(µg/m**<sup>**3**</sup>**)**|**MAE**<br>**(µg/m**<sup>**3**</sup>**)**|**R**|**RMSE**<br>**(µg/m**<sup>**3**</sup>**)**|**MAE**<br>**(µg/m**<sup>**3**</sup>**)**|
|RF|0.837|15.575|10.954|0.641|18.266|13.077|0.649|17.027|12.409|
|XGB|0.842|14.917|10.402|0.642|18.389|13.095|0.653|17.000|12.388|
|CB|0.819|16.164|11.479|**0.643**|**18.193**|**12.954**|**0.665**|**16.582**|**12.084**|



_Appl. Sci._ **2025** , _15_ , 10997 

9 of 19 

Considering the results given in Table 2, all the results obtained were lower than the standard deviations of the data itself (34.16 µg/m<sup>3</sup> for training, 31.40 µg/m<sup>3</sup> for validation, and 29.86 µg/m<sup>3</sup> for testing). This shows that all algorithms performed well. Although the error values of the models established with Geos-CF data in the training phase were lower than those of the models established with Sentinel-5P data, the RMSE and MAE of the models established with S5P in the validation and testing phases were relatively lower, and the R was higher. 

Although XGB gave the highest R (0.945 and 0.842) and lowest RMSE (9.157 and 14.917 µg/m<sup>3</sup> ) and MAE (6.405 and 10.402 µg/m<sup>3</sup> ) values in the training phase, it obtained the lowest R and highest error values in the validation and testing phases for both S5P and Geos-CF data due to its tendency to overfitting. CB gave the best results in the validation and testing phases in both data models due to its generalization capability. When ranked in terms of performance, the CB algorithm is followed by RF and XGB, respectively. 

The station-based diagrams, including accuracy metrics (RMSE, MAE, and R) for all algorithms, are presented in Figure 4. In general, the RF model (Figure 4a,b) and CB model (Figure 4e,f) exhibited similar performance for the Sentinel-5P dataset, characterized by high correlation coefficients (R > 0.7) and relatively low RMSE (mostly below 15 µg/m<sup>3</sup> ). The color distribution further indicates lower MAE values, suggesting that RF and CB achieved more stable and accurate estimations across different stations. The XGB model (Figure 4c,d) showed a wider spread of RMSE values, with several points extending toward higher error levels, reflecting greater variability in prediction accuracy. The CB model (Figure 4e,f) demonstrated consistent and balanced results, with moderate RMSE and relatively high correlation values similar to those of RF. Across all models, the Sentinel-5P-based results (Figure 4b,d,f) generally outperform those derived from Geos-CF (Figure 4a,c,e), indicating a better agreement between Sentinel-5P observations and the model outputs. Overall, the station-based analysis confirms that RF and CB provided the most robust and reliable predictions, while Sentinel-5P data provided stronger consistency with the modeled variables. Additionally, the analysis revealed that the models consistently exhibit high errors at the same stations. The three stations with the highest errors exceeding 20 µg/m<sup>3</sup> (Avcılar, Ümraniye2, and Esenler) are located in spatially distinct areas within the most densely populated districts. This may be attributed to abrupt fluctuations in measured values, which could have increased the model errors at these stations. 

In addition to the quantitative evaluation, a qualitative assessment is also important in assessing the model’s performance. For this purpose, seasonal thematic maps of NO2 distribution were created with seasonal average data for 2024. Maps created with Sentinel5P are given in Figure 5, while maps created with Geos-CF are given in Figure 6. 

According to Figure 5, seasonal maps obtained from the XGB model are quite noisy, unlike those from other models. It was observed that the amount of NO2 was high in the northern parts of the city where forested areas were dense. Seasonal variation in NO2 distribution shows significant changes in both the RF and CB models. During winter and spring, NO2 levels are higher in the southern parts of the Bosphorus, where urban areas are dense, while they are lower in other areas. While NO2 levels decrease in the summer months, they increase again in the autumn. Differences between the two model results are particularly evident in summer and autumn. The sharp linear changes in the results are due to the pixel sizes of the Geos-CF data. 

When the thematic maps obtained with the models created with Geos-CF are examined, the XGB model has a high noise level, similar to the results obtained with the Sentinel-5P models. RF and CB models also show consistent seasonal distribution. Model results also diverge between summer and autumn seasons. 

_Appl. Sci._ **2025** , _15_ , 10997 

10 of 19 

In order to evaluate the accuracy of the produced thematic maps, an accuracy assessment was conducted using seasonal average values obtained from the stations. The calculated R, RMSE, and MAE values for four seasons are given in Figure 7. 



<!-- Start of picture text -->
a) b)<br>00 7on Sentinel-5P omoz Geos-CF 28<br>043 — 0.43<br>2<br>on oR 062 oR<br>@ e @<br>o7e 078 os<br>°°<br>°<br>ee Z<br>e 0.90 e 0.90 a8<br>.. e *,° \ °,e° 8%$<br>4 \ ox - 3 \or fie<br>ee oe \\ oe \|<br>1.00 1.00 bd<br>os 0 1% 2 2 % % oO 5 1 1 0 2 0 3%<br>RMSE RMSE<br>4<br>°) qd)<br>000 Sentinel-5P 0.00 Geos-CF 28<br>oz 0<br>TS 0.43, ~~ 043<br>° 24<br>oe ok oe<br>e<br>078 2 078 20<br>ry PY 4<br>oeme oePYoe 0.90 eoeee@e @ 0.90 16 3<br>"<br>22° os? 4 8b \\osr J?<br>So \ - v e \\<br>1.00 1.00, i<br>o 5 © 1% 2 2% % % o 5 0 6% 2 2% 0 3<br>RMSE RMSE<br>4<br>e) f)<br>0.00 0:022 Sentinel-5P 0.0002oz Geos-CF 28<br>043 ous<br>0.62 R 0.62 R 2<br>e °<br>| 078 e 0.78 20<br>e bf y<br>ee ° FE<br>° ° 600 oe°, 0.90 16<br>cee e @ \<br>eee see \\<br>%2,%e \ oor erea \\ oar 2<br>Pia \ ee \<br>e \ \<br>1.00 1.00 8<br>o 5 0 % o 8 0 8 OF 5 0 8 2 2 0 95<br>RMSE RMSE<br>4<br><!-- End of picture text -->

**Figure 4.** Station-based RMSE, MAE, and R diagrams for 2024. ( **a** , **b** ): RF, ( **c** , **d** ): XGB, and ( **e** , **f** ): CB. 

_Appl. Sci._ **2025** , _15_ , 10997 

11 of 19 



<!-- Start of picture text -->
RF XGB cB<br>A<br>a<br>2 ee ete* ies =~: peeBS Be Se<br>Li, aS as<br>ary Bae ay os rs hay<br>A A A<br>ie. sa Te EE MERI<br>[2p Be ne<br><7 asi ge GREY Ti tae ee<br>NO2 [g/m] MMM 0-20 [T) 20-30 [| 30-40 [I 40-50 MM >50<br><!-- End of picture text -->

**Figure 5.** Seasonal maps created with the Sentinel-5P seasonal average dataset for 2024. 



<!-- Start of picture text -->
RF XGB cB<br>SS rs meat SiS ex ee ype<br>Ase ” i Eh.<br>=. Bain... Te<br>A iS a<br>Se BBs a<br>2 Emme ey tgs Bee ee. ers<br>NO2[ug/m’] MMMM 0-20 [7] 20-30 | | 30-40 [| 40-50 MM >50<br><!-- End of picture text -->

**Figure 6.** Seasonal maps created with the Geos-CF seasonal average dataset for 2024. 

Considering Figure 7, among the models built with Sentinel-5P, XGB has the lowest R and highest RMSE and MAE errors in all seasons. After both quantitative and qualitative evaluations, the XGB algorithm was found to be unsuccessful on this dataset. In the RF and CB models, RF was relatively successful only in the summer season, while CB performed successfully in other seasons. When the error amounts were examined, it was determined that the models’ error amounts were higher, and the correlations were lower in the spring and autumn seasons compared to the summer and winter seasons. 

_Appl. Sci._ **2025** , _15_ , 10997 

12 of 19 



<!-- Start of picture text -->
Sentinel-5P Geos-CF<br>oso 080<br>a7080 070060<br>050 050<br>0030 00020<br>020oxo 020a0<br>on ‘Winter ‘Spring “Summer “Autarnn 9.00 ‘Winter ‘Spring: ‘Summer ‘Auture,<br>snr oss ass oe wen a0 cor oer<br>txes as oa oat a3 Tics ost 930 oo ase<br>sce ae o58 oa: an sce er a5 ost oer<br>Fa Fa<br>2s Pa<br>5?Zs i »18<br>s5<br>2 0°‘Winter ‘Spring ‘Sumener “Autumn 2 0° ‘Winter| ‘Spring ‘Summer “Autumn<br><!-- End of picture text -->



<!-- Start of picture text -->
Fa Fa<br>2» 2»<br>i 15 I 18<br>ge z*<br>55<br>°<br>on Mine So Sime tune ° wn See ‘Sune tune<br>sxe aes wor ‘eat ‘eo ous ‘oat ‘at om<br>sce nse 223 te38 ne tres th38 ma ‘a8 ‘eer<br>hae ites teat on sce ua toe aa wt<br><!-- End of picture text -->

**Figure 7.** Accuracy assessment of seasonal maps using R, RMSE, and MAE. 

Among models built with Geos-CF data, XGB appears to be the most successful model with the lowest error values in the quantitative evaluation, even though it fails in the qualitative assessment. This demonstrates that quantitative assessments alone are not sufficient to evaluate ML model results. Among the RF and CB models, CB performed well in all seasons except summer. Spring and autumn were also the seasons with the highest error rates. 

When the Sentinel-5P and Geos-CF model results are compared, it is observed that the RF and CB model results shown in Figures 5 and 6 mostly yield similar results. In the quantitative evaluation, it was determined that the CB model created with Sentinel-5P data showed better results than the models established with Geos-CF in both the training, validation, and testing stages (Table 2) and in the seasonal evaluation analyses (Figure 7). 

## **5. Discussion** 

In the study, ground-level NO2 estimation analysis of Istanbul province was carried out using three different ML algorithms (RF, XGB, and CB) and two different datasets, including satellite-derived NO2 (Sentinel-5P and Geos-CF), meteorological, environmental, and social factors. The model created using data collected between 2019 and 2023 was tested with data from 2024, and its performance was compared both quantitatively and qualitatively. While many studies indicate that the XGB algorithm performs well in estimating ground-level NO2 [13,17,27,58], in this study, XGB exhibited the lowest performance among the three algorithms. Shetty et al. (2024) [22] also reported that the model performed poorly in Türkiye in their study using the XGB algorithm over Europe. While CB produced the best results among the three algorithms, Figure 8 presents the station-based relative error distribution, calculated using both seasonal station observations and the estimated values. 

_Appl. Sci._ **2025** , _15_ , 10997 

13 of 19 



<!-- Start of picture text -->
&<br>2 w<br>&<br>toy hf ofl<br>° iil<br>As}<br>- Pd.i . rare<br>—<br>WS Winter WE Spring MO Summer WE Autumn<br>S ee<br>§ pat<br>iil<br>So<br>- PdCope. Am<br>¢ —_—<br><!-- End of picture text -->

**Figure 8.** Relative error distribution of seasons for ( **a** ) Sentinel-5P and ( **b** ) Geos-CF. 

The highest errors were recorded in spring in the southern parts of the Bosphorus, whereas in summer, errors were more pronounced in the northern areas compared to other seasons. Error levels were generally higher at stations on the Asian side (east of the Bosphorus) than on the European side (west of the Bosphorus), with the lowest errors observed in winter at stations in the western part of the city. This is primarily due to the lack of sufficient stations in the northern part of the Asian side. It is noteworthy that the distribution of relative errors is quite similar in both datasets. 

To investigate the impacts of features on the estimation of ground-level NO2, SHAP analysis was performed for each data model. The results are presented in Figure 9, and the 20 features are listed in order of importance. While Sentinel-5P NO2 data became prominent as the most important factor, Geos-CF data contributed to the model as the fourth most important factor. This reveals that location features (RL, PD, and DEM) are more important than Geos-CF data. It also demonstrated the necessity of including satellite-derived NO2 data in the ground-level NO2 estimation model [58]. 

_Appl. Sci._ **2025** , _15_ , 10997 

14 of 19 



<!-- Start of picture text -->
Sentinel-5P Geos-CF<br>590.NOQ QA- = RL St<br>RL << > apa<br>PO i oo DEM OGD ce<br>Dem <— Geos_No2 l———-<br>za =e Za =<br>nm > viom ->se<br>viom oe NTL ee<br>novi _— novi 8E5-—<br>viomae <Q>—. Ps§ s00_oustnom Oe— | 35<br>0D DUST — ‘ow por Eee ow g<br>a eo mis + —<br>rerec +> aH ~<br>Dor —_ TPREC _—<br>1s + 1s rs<br>eM — 1M +<br>is + ap +<br>Q Ps.<br>:  ¢ ¢<br>ao 5 0 5 b 6 0 @ to 5 0 5 » 8 m ®<br>‘SHAP value (impact on model output) SHAP value (impact on model output)<br><!-- End of picture text -->

**Figure 9.** SHAP analysis results for the CB model estimated for both Sentinel-5P and Geos-CF. 

In both models, RL and PD are among the top three most important factors. High RL and PD values increase the model output. Dense populations and road networks are strongly associated with higher surface NO2 levels. The distribution of road networks and PD is given in Figure 10a,b. It is observed that the road network is dense in the southern parts of the Bosphorus where the population is concentrated. The CB model results in Figures 5 and 6 show that ground-level NO2 values are high in areas where the road network and PD are dense. The main reason for this situation is vehicle emissions and human activities that cause NO2 formation [59]. Shao et al. (2023) [27] revealed that urbanization and population growth exhibit a power law relationship with NO2 concentration. In addition, the seasonal variation in ground-level NO2 concentration in this region can also be associated with human activities. The fact that NO2 levels are high in winter and spring, decrease in summer, and increase again in autumn is due to heating activities in densely populated areas [60]. 

DEM was identified as the third most important factor after RL and PD, and its distribution in Istanbul is given in Figure 10c. According to the SHAP results, decreasing elevation causes an increase in ground-level NO2. The reason for this can be the high settlement and human population on the shores of the Bosphorus, where the elevation is low. When Figure 10 is examined, in areas with high elevation, both the road network and population density decrease, and therefore the NO2 level also decreases. 

The following five important features are ZL, NTL, V_10M, NDVI, and ZPBL. While high ZL, NTL, and V_10M values cause an increase in ground-level NO2, low NDVI and ZPBL contribute to this rise. High NTL and low NDVI can also be associated with urbanization [22]. In densely urbanized areas, NTL is higher and NDVI is lower. Since wind is an effective parameter in NO2 transport [28], it was found in this study that northward wind (V_10M) is more effective than eastward wind (U_10M). Low ZPBL values may be associated with increased ground-level NO2 values, as air pollutants tend to concentrate at low altitudes near the Earth’s surface [61]. Low PBLH also causes increased groundlevel NO2 concentrations, especially in coastal areas [62]. Meteorological parameters were 

_Appl. Sci._ **2025** , _15_ , 10997 

15 of 19 

ranked among the last ten least influential features within the scope of the study, lagging behind human activities and topography in estimating ground-level NO2. 



<!-- Start of picture text -->
> re<br>ee ;<br>= SE ARS ne 3 A<br>[o)|<br>°.<br><!-- End of picture text -->

**Figure 10.** The most effective features: ( **a** ) Road network, ( **b** ) Population density, and ( **c** ) Elevation. 

## **6. Strengths and Limitations** 

The strengths and limiting factors of this study were identified. The strength of the study is the successful demonstration of ground-level NO2 distribution over Istanbul using data with varying spatial resolutions. The CB models built with two different datasets (Sentinel-5P and Geos-CF) produced both quantitative and qualitative results, but the model built with Sentinel-5P data performed better, thanks to its higher spatial resolution. 

The limitation of the study is the lack of both in situ and auxiliary data. When the distribution of ground air quality monitoring stations in Figure 1 is examined, it is observed that the stations are not distributed homogeneously throughout the city. Because NO2 emissions are generated by vehicle emissions, industrial activities, and human activities, terrestrial air quality monitoring stations are located in densely populated areas. However, this poses a limitation in regional ground-level NO2 estimation analyses. 

Another limitation is that meteorological data with a spatial resolution of approximately 11 km, such as ERA5-Land, cannot be used within the scope of this study. This is because water areas are masked in this dataset. Because Istanbul is geographically surrounded by the Black Sea and the Sea of Marmara, the absence of pixels corresponding to terrestrial air quality measurement stations on land results in data loss due to pixel size. Therefore, Geos-CF data were used instead of ERA5-Land data. Studies frequently utilize ERA5 products and perform analyses at a broader spatial scale [23,26]. Future studies will test the model with ERA5 data at the same spatial resolution as Geos-CF, and analyses will be expanded to include other metropolitan cities. Furthermore, adding traffic data in cities with high human activity will also increase model accuracy. Additionally, the models were analyzed on a seasonal basis in this study. Future research could extend this approach to monthly assessments; however, producing results at a daily temporal resolution is not feasible due to data gaps. 

## **7. Conclusions** 

In this study, ground-level NO2 estimation was performed using three different ML algorithms and two different datasets. In addition to determining the most accurate 

_Appl. Sci._ **2025** , _15_ , 10997 

16 of 19 

model, each model incorporated 20 features, and their relative contributions were assessed through SHAP analysis. The CB model was identified as the most successful. Although the atmospheric GEOS-CF data with lower spatial resolution influenced the visual outcomes in models based on both Sentinel-5P and GEOS-CF inputs, the models were still able to accurately capture the spatial and temporal distribution of ground-level NO2 both quantitatively and qualitatively. 

The results highlight the critical role of anthropogenic indicators (e.g., PD, road networks, NTL), topographic factors (DEM), and air pollution variables (NO2 from Sentinel-5P and GEOS-CF) in driving model performance, while meteorological factors contributed secondarily. In future studies, we will expand the data pool by incorporating traffic data and various atmospheric datasets and will perform more comprehensive analyses using deep learning models to enhance model accuracy. 

**Funding:** This research received no external funding. 

**Institutional Review Board Statement:** Not applicable. 

**Informed Consent Statement:** Not applicable. 

**Data Availability Statement:** The data presented in this study are available on request from the corresponding author (the data are not publicly available due to privacy or ethical restrictions). 

**Acknowledgments:** The author expresses her gratitude to the Istanbul Metropolitan Municipality for providing the air quality ground monitoring station data used in this study. 

**Conflicts of Interest:** The author declares no conflicts of interest. 

## **Abbreviations** 

The following abbreviations are used in this manuscript: 

NO2 Nitrogen dioxide ML Machine Learning RF Random Forest Regression XGB XGBoost Regression CB CatBoost Regression XAI Explainable Artificial Intelligence SHAP Shapley Additive exPlanations 

## **References** 

1. Liu, F.; Beirle, S.; Zhang, Q.; Dörner, S.; He, K.; Wagner, T. NOx Lifetimes and Emissions of Cities and Power Plants in Polluted Background Estimated by Satellite Observations. _Atmos. Chem. Phys._ **2016** , _16_ , 5283–5298. [CrossRef] 

2. Khan, R.R.; Siddiqui, M.J. Review on Effects of Particulates: Sulfur Dioxide and Nitrogen Dioxide on Human Health. _Int. Res. J. Environ. Sci._ **2014** , _3_ , 70–73. 

3. Eum, K.-D.; Kazemiparkouhi, F.; Wang, B.; Manjourides, J.; Pun, V.; Pavlu, V.; Suh, H. Long-Term NO2 Exposures and CauseSpecific Mortality in American Older Adults. _Environ. Int._ **2019** , _124_ , 10–15. [CrossRef] 

4. Manisalidis, I.; Stavropoulou, E.; Stavropoulos, A.; Bezirtzoglou, E. Environmental and Health Impacts of Air Pollution: A Review. _Front. Public Health_ **2020** , _8_ , 505570. [CrossRef] 

5. Seinfeld, J.H.; Pandis, S.N. _Atmospheric Chemistry and Physics: From Air Pollution to Climate Change_ ; John Wiley & Sons: Hoboken, NJ, USA, 2016. 

6. Chen, T.-M.; Kuschner, W.G.; Gokhale, J.; Shofer, S. Outdoor Air Pollution: Nitrogen Dioxide, Sulfur Dioxide, and Carbon Monoxide Health Effects. _Am. J. Med. Sci._ **2007** , _333_ , 249–256. [CrossRef] 

7. Cao, H.; Han, L. The Short-Term Impact of the COVID-19 Epidemic on Socioeconomic Activities in China Based on the OMI-NO2 Data. _Environ. Sci. Pollut. Res._ **2022** , _29_ , 21682–21691. [CrossRef] 

8. Schneider, P.; Lahoz, W.A.; van der A, R. Recent Satellite-Based Trends of Tropospheric Nitrogen Dioxide over Large Urban Agglomerations Worldwide. _Atmos. Chem. Phys._ **2015** , _15_ , 1205–1220. [CrossRef] 

_Appl. Sci._ **2025** , _15_ , 10997 

17 of 19 

9. Güçlü, Y.S.; Dabanlı, I.;<sup>˙</sup> ¸Si¸sman, E.; ¸Sen, Z. Air Quality (AQ) Identification by Innovative Trend Diagram and AQ Index Combinations in Istanbul Megacity. _Atmos. Pollut. Res._ **2019** , _10_ , 88–96. [CrossRef] 

10. Wang, W.; Li, B.; Chen, B. Improved Surface NO2 Retrieval: Double-Layer Machine Learning Model Construction and SpatioTemporal Characterization Analysis in China (2018–2023). _J. Environ. Manag._ **2025** , _384_ , 125439. [CrossRef] 

11. Zhang, Y.; Li, Z.; Wei, J.; Zhan, Y.; Liu, L.; Yang, Z.; Zhang, Y.; Liu, R.; Ma, Z. Long-Term Exposure to Ambient NO2 and Adult Mortality: A Nationwide Cohort Study in China. _J. Adv. Res._ **2022** , _41_ , 13–22. [CrossRef] 

12. Zhang, D.; Shi, R.; Zhou, Y.; Zheng, L.; Chen, M. The Spatial Distribution Characteristics and Ground-Level Estimation of NO2 and SO2 over Huaihe River Basin and Shanghai Based on Satellite Observations. In Proceedings of the Remote Sensing and Modeling of Ecosystems for Sustainability XV, San Diego, CA, USA, 19–23 August 2018; Gao, W., Chang, N.-B., Wang, J., Eds.; SPIE: Bellingham, WA, USA, 2018; p. 22. 

13. Kang, Y.; Choi, H.; Im, J.; Park, S.; Shin, M.; Song, C.-K.; Kim, S. Estimation of Surface-Level NO2 and O3 Concentrations Using TROPOMI Data and Machine Learning over East Asia. _Environ. Pollut._ **2021** , _288_ , 117711. [CrossRef] 

14. Fernandes, A.P.; Riffler, M.; Ferreira, J.; Wunderle, S.; Borrego, C.; Tchepel, O. Spatial Analysis of Aerosol Optical Depth Obtained by Air Quality Modelling and SEVIRI Satellite Observations over Portugal. _Atmos. Pollut. Res._ **2019** , _10_ , 234–243. [CrossRef] 

15. Duncan, B.N.; Prados, A.I.; Lamsal, L.N.; Liu, Y.; Streets, D.G.; Gupta, P.; Hilsenrath, E.; Kahn, R.A.; Nielsen, J.E.; Beyersdorf, A.J.; et al. Satellite Data of Atmospheric Pollution for U.S. Air Quality Applications: Examples of Applications, Summary of Data End-User Resources, Answers to FAQs, and Common Mistakes to Avoid. _Atmos. Environ._ **2014** , _94_ , 647–662. [CrossRef] 

16. Qin, K.; Rao, L.; Xu, J.; Bai, Y.; Zou, J.; Hao, N.; Li, S.; Yu, C. Estimating Ground Level NO2 Concentrations over Central-Eastern China Using a Satellite-Based Geographically and Temporally Weighted Regression Model. _Remote Sens._ **2017** , _9_ , 950. [CrossRef] 

17. Chi, Y.; Fan, M.; Zhao, C.; Yang, Y.; Fan, H.; Yang, X.; Yang, J.; Tao, J. Machine Learning-Based Estimation of Ground-Level NO2 Concentrations over China. _Sci. Total Environ._ **2022** , _807_ , 150721. [CrossRef] 

18. Bahadur, F.T.; Shah, S.R.; Nidamanuri, R.R. Applications of Remote Sensing Vis-à-Vis Machine Learning in Air Quality Monitoring and Modelling: A Review. _Environ. Monit. Assess._ **2023** , _195_ , 1502. [CrossRef] [PubMed] 

19. Fu, J.; Tang, D.; Grieneisen, M.L.; Yang, F.; Yang, J.; Wu, G.; Wang, C.; Zhan, Y. A Machine Learning-Based Approach for Fusing Measurements from Standard Sites, Low-Cost Sensors, and Satellite Retrievals: Application to NO2 Pollution Hotspot Identification. _Atmos. Environ._ **2023** , _302_ , 119756. [CrossRef] 

20. Cedeno Jimenez, J.R.; Pugliese Viloria, A.d.J.; Brovelli, M.A. Estimating Daily NO2 Ground Level Concentrations Using Sentinel5P and Ground Sensor Meteorological Measurements. _ISPRS Int. J. Geoinf._ **2023** , _12_ , 107. [CrossRef] 

21. Yagmur Aydin, N.; Aydin, I. Estimation of Ground-Level NO2 Concentrations over Megacities Using Sentinel-5P and Machine Learning Models: A Case Study of Istanbul. _Int. Arch. Photogramm. Remote Sens. Spat. Inf. Sci._ **2025** , _XLVIII-M-6–2025_ , 303–308. [CrossRef] 

22. Shetty, S.; Schneider, P.; Stebel, K.; David Hamer, P.; Kylling, A.; Koren Berntsen, T. Estimating Surface NO2 Concentrations over Europe Using Sentinel-5P TROPOMI Observations and Machine Learning. _Remote Sens. Environ._ **2024** , _312_ , 114321. [CrossRef] 

23. Griffin, D.; Hempel, C.; McLinden, C.; Kharol, S.K.; Lee, C.; Fogal, A.; Sioris, C.; Shephard, M.; You, Y. Development and Validation of Satellite-Derived Surface NO2 Estimates Using Machine Learning versus Traditional Approaches in North America. _EGUsphere_ **2025** , _2025_ , 1–20. [CrossRef] 

24. Araki, S.; Shima, M.; Yamamoto, K. Spatiotemporal Land Use Random Forest Model for Estimating Metropolitan NO2 Exposure in Japan. _Sci. Total Environ._ **2018** , _634_ , 1269–1277. [CrossRef] 

25. Chan, K.L.; Khorsandi, E.; Liu, S.; Baier, F.; Valks, P. Estimation of Surface NO2 Concentrations over Germany from TROPOMI Satellite Observations Using a Machine Learning Method. _Remote Sens._ **2021** , _13_ , 969. [CrossRef] 

26. Long, S.; Wei, X.; Zhang, F.; Zhang, R.; Xu, J.; Wu, K.; Li, Q.; Li, W. Estimating Daily Ground-Level NO2 Concentrations over China Based on TROPOMI Observations and Machine Learning Approach. _Atmos. Environ._ **2022** , _289_ , 119310. [CrossRef] 

27. Shao, Y.; Zhao, W.; Liu, R.; Yang, J.; Liu, M.; Fang, W.; Hu, L.; Adams, M.; Bi, J.; Ma, Z. Estimation of Daily NO2 with Explainable Machine Learning Model in China, 2007–2020. _Atmos. Environ._ **2023** , _314_ , 120111. [CrossRef] 

28. Sun, W.; Tack, F.; Clarisse, L.; Schneider, R.; Stavrakou, T.; Van Roozendael, M. Inferring Surface NO2 over Western Europe: A Machine Learning Approach with Uncertainty Quantification. _J. Geophys. Res. Atmos._ **2024** , _129_ , e2023JD040676. [CrossRef] 

29. Lundberg, S.; Lee, S.-I. A Unified Approach to Interpreting Model Predictions. _arXiv_ **2017** , arXiv:1705.07874. [CrossRef] 

30. Khorrami, B.; Heidarlou, H.B.; Feizizadeh, B. Evaluation of the Environmental Impacts of Urbanization from the Viewpoint of Increased Skin Temperatures: A Case Study from Istanbul, Turkey. _Appl. Geomat._ **2021** , _13_ , 311–324. [CrossRef] 

31. Bozkurt, S.G.; Ku¸sak, L. Detection of Population Density, LULC Variation and Cross-Regional Similarities Using K-Means Clustering Algorithm in Istanbul Example. _Mimar. Bilim. Uygulamaları Derg._ **2024** , _9_ , 69–86. [CrossRef] 

32. Akın, A.; Sunar, F.; Berbero˘glu, S. Urban Change Analysis and Future Growth of Istanbul. _Environ. Monit. Assess._ **2015** , _187_ , 506. [CrossRef] 

33. Sentinel-5P OFFL NO2: Offline Nitrogen Dioxide. Available online: https://developers.google.com/earth-engine/datasets/ catalog/COPERNICUS_S5P_OFFL_L3_NO2 (accessed on 1 August 2025). 

_Appl. Sci._ **2025** , _15_ , 10997 

18 of 19 

34. SRTM. Available online: https://www.earthdata.nasa.gov/data/instruments/srtm (accessed on 1 August 2025). 

35. Rouse, J.W., Jr.; Haas, R.H.; Schell, J.A.; Deering, D.W. _Monitoring the Vernal Advancement and Retrogradation (Green Wave Effect) of Natural Vegetation_ ; NASA: Washington, DC, USA, 1973. 

36. MODIS. Available online: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD43A4#description (accessed on 1 August 2025). 

37. Elvidge, C.D.; Baugh, K.; Zhizhin, M.; Hsu, F.C.; Ghosh, T. VIIRS Night-Time Lights. _Int. J. Remote Sens._ **2017** , _38_ , 5860–5879. [CrossRef] 

38. Levin, N.; Kyba, C.C.M.; Zhang, Q.; Sánchez de Miguel, A.; Román, M.O.; Li, X.; Portnov, B.A.; Molthan, A.L.; Jechow, A.; Miller, S.D.; et al. Remote Sensing of Night Lights: A Review and an Outlook for the Future. _Remote Sens. Environ._ **2020** , _237_ , 111443. [CrossRef] 

39. VIIRS Lunar Gap-Filled BRDF Nighttime Lights. Available online: https://developers.google.com/earth-engine/datasets/ catalog/NASA_VIIRS_002_VNP46A2#description (accessed on 1 August 2025). 

40. Geos-CF. Available online: https://developers.google.com/earth-engine/datasets/catalog/NASA_GEOS-CF_v1_rpl_tavg1hr# description (accessed on 1 August 2025). 

41. Liu, N.; Lin, W.; Ma, J.; Xu, W.; Xu, X. Seasonal Variation in Surface Ozone and Its Regional Characteristics at Global Atmosphere Watch Stations in China. _J. Environ. Sci._ **2019** , _77_ , 291–302. [CrossRef] [PubMed] 

42. Qin, K.; Han, X.; Li, D.; Xu, J.; Loyola, D.; Xue, Y.; Zhou, X.; Li, D.; Zhang, K.; Yuan, L. Satellite-Based Estimation of Surface NO2 Concentrations over East-Central China: A Comparison of POMINO and OMNO2d Data. _Atmos. Environ._ **2020** , _224_ , 117322. [CrossRef] 

43. OPTUNA. Available online: https://optuna.org/ (accessed on 1 August 2025). 

44. Breiman, L. Random Forests. _Mach. Learn._ **2001** , _45_ , 5–32. [CrossRef] 

45. Chen, J.; Zhu, S.; Wang, P.; Zheng, Z.; Shi, S.; Li, X.; Xu, C.; Yu, K.; Chen, R.; Kan, H.; et al. Predicting Particulate Matter, Nitrogen Dioxide, and Ozone across Great Britain with High Spatiotemporal Resolution Based on Random Forest Models. _Sci. Total Environ._ **2024** , _926_ , 171831. [CrossRef] 

46. Vaishnavi, K.; Sreya, G.; Reddy, K.K.; P R, A. Machine Learning for Air Quality Prediction: Random Forest Classifier. In Proceedings of the 2024 Fourth International Conference on Advances in Electrical, Computing, Communication and Sustainable Technologies (ICAECT), Bhilai, India, 11–12 January 2024; pp. 1–5. 

47. Sharda, S.; Kumar, S.; Setia, R.; Dhiman, P.; Patel, N.R.; Pateriya, B.; Salem, A.; Elbeltagi, A. Evaluation of Different Spectral Indices for Wheat Lodging Assessment Using Machine Learning Algorithms. _Sci. Rep._ **2025** , _15_ , 21774. [CrossRef] 

48. Chen, T.; Guestrin, C. XGBoost. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, 13–17 August 2016; ACM: New York, NY, USA, 2016; pp. 785–794. 

49. Ozturk, M.Y.; Colkesen, I. A Novel Hybrid Methodology Integrating Pixel- and Object-Based Techniques for Mapping Land Use and Land Cover from High-Resolution Satellite Data. _Int. J. Remote Sens._ **2024** , _45_ , 5640–5678. [CrossRef] 

50. Georganos, S.; Grippa, T.; Vanhuysse, S.; Lennert, M.; Shimoni, M.; Wolff, E. Very High Resolution Object-Based Land Use–Land Cover Urban Classification Using Extreme Gradient Boosting. _IEEE Geosci. Remote Sens. Lett._ **2018** , _15_ , 607–611. [CrossRef] 

51. Rumora, L.; Miler, M.; Medak, D. Impact of Various Atmospheric Corrections on Sentinel-2 Land Cover Classification Accuracy Using Machine Learning Classifiers. _ISPRS Int. J. Geoinf._ **2020** , _9_ , 277. [CrossRef] 

52. Abdi, A.M. Land Cover and Land Use Classification Performance of Machine Learning Algorithms in a Boreal Landscape Using Sentinel-2 Data. _GIScience Remote Sens._ **2020** , _57_ , 1–20. [CrossRef] 

53. Dorogush, A.V.; Ershov, V.; Gulin, A. CatBoost: Gradient Boosting with Categorical Features Support. _arXiv_ **2018** , arXiv:1810.11363. [CrossRef] 

54. Kulkarni, C.S. Advancing Gradient Boosting: A Comprehensive Evaluation of the CatBoost Algorithm for Predictive Modeling. _J. Artif. Intell. Mach. Learn. Data Sci._ **2022** , _1_ , 54–57. [CrossRef] 

55. Pham, T.D.; Yokoya, N.; Nguyen, T.T.T.; Le, N.N.; Ha, N.T.; Xia, J.; Takeuchi, W.; Pham, T.D. Improvement of Mangrove Soil Carbon Stocks Estimation in North Vietnam Using Sentinel-2 Data and Machine Learning Approach. _GIScience Remote Sens._ **2021** , _58_ , 68–87. [CrossRef] 

56. Ozturk, M.Y.; Colkesen, I. Development of Transferable Hybrid Deep Learning Networks for Temporal and Multi-Regional Mapping of Poplar Plantations with Sentinel-2. _Adv. Space Res._ **2025** , _76_ , 4249–4279. [CrossRef] 

57. Lundberg, S.M.; Erion, G.; Chen, H.; DeGrave, A.; Prutkin, J.M.; Nair, B.; Katz, R.; Himmelfarb, J.; Bansal, N.; Lee, S.-I. From Local Explanations to Global Understanding with Explainable AI for Trees. _Nat. Mach. Intell._ **2020** , _2_ , 56–67. [CrossRef] [PubMed] 

58. Wei, Q.; Song, W.; Dai, B.; Wu, H.; Zuo, X.; Wang, J.; Chen, J.; Li, J.; Li, S.; Chen, Z. Spatiotemporal Estimation of Surface NO2 Concentrations in the Pearl River Delta Region Based on TROPOMI Data and Machine Learning. _Atmos. Pollut. Res._ **2025** , _16_ , 102353. [CrossRef] 

59. Kang, H.; Zhu, B.; Zhu, C.; de Leeuw, G.; Hou, X.; Gao, J. Natural and Anthropogenic Contributions to Long-Term Variations of SO2, NO2, CO, and AOD over East China. _Atmos. Res._ **2019** , _215_ , 284–293. [CrossRef] 

_Appl. Sci._ **2025** , _15_ , 10997 

19 of 19 

60. Yu, S.; Yin, S.; Zhang, R.; Wang, L.; Su, F.; Zhang, Y.; Yang, J. Spatiotemporal Characterization and Regional Contributions of O3 and NO2: An Investigation of Two Years of Monitoring Data in Henan, China. _J. Environ. Sci._ **2020** , _90_ , 29–40. [CrossRef] 

61. Xiao, K.; Wang, Y.; Wu, G.; Fu, B.; Zhu, Y. Spatiotemporal Characteristics of Air Pollutants (PM10, PM2.5, SO2, NO2, O3, and CO) in the Inland Basin City of Chengdu, Southwest China. _Atmosphere_ **2018** , _9_ , 74. [CrossRef] 

62. Lee, S.-J.; Lee, J.; Greybush, S.J.; Kang, M.; Kim, J. Spatial and Temporal Variation in PBL Height over the Korean Peninsula in the KMA Operational Regional Model. _Adv. Meteorol._ **2013** , _2013_ , 1–16. [CrossRef] 

**Disclaimer/Publisher’s Note:** The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content. 

