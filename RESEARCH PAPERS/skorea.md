Atmospheric Environment 331 (2024) 120603 



<!-- Start of picture text -->
pO Eat<br>ELSEVIER<br><!-- End of picture text -->

Contents lists available at ScienceDirect 

# Atmospheric Environment 

journal homepage: www.elsevier.com/locate/atmosenv 



<!-- Start of picture text -->
sp<br>=, |<br><!-- End of picture text -->

Ozone, nitrogen dioxide, and PM2.5 estimation from observation-model machine learning fusion over S. Korea: Influence of observation density, chemical transport model resolution, and geostationary remotely sensed AOD 



<!-- Start of picture text -->
®<br><!-- End of picture text -->

Beiming Tang<sup>a,*</sup> , Charles O. Stanier<sup>b,c</sup> , Gregory R. Carmichael<sup>b</sup> , Meng Gao<sup>d</sup> 

a _National Oceanic and Atmospheric Administration (NOAA) Air Resources Laboratory (ARL) & George Mason University, USA_ b _The University of Iowa, Department of Chemical and Biochemical Engineering, USA_ c _The University of Iowa, IIHR Hydroscience & Engineering, USA_ d _Hong Kong Baptist University, Department of Geography, China_ 

H I G H L I G H T S 

- PM2.5, ozone, and NO2 machine learning (random forest) retrospective data fusion and downscaling at 1 × 1 km<sup>2</sup> over S. Korea. 

- Good performance (R 0.73–0.95), but decreases if less than three ground sites per million people. 

- Coarse spatial resolution global reanalysis fields are widely available and aid in machine learning fusion. 

A R T I C L E I N F O A B S T R A C T _Keywords:_ High-resolution multi-component estimates of ground-level air pollutants are necessary for assessing their imHigh-resolution exposure modeling pacts to human health, agriculture, and ecosystems. We demonstrate a high-resolution fusion and downscaling Machine learning approach over South Korea for May 2016 and May 2021. Daily 1 km fine particulate matter (PM2.5), ozone (O3), Multiple species air pollution and nitrogen dioxide (NO2) concentrations are calculated at ground level using a random forest machine learning GEMS (ML) algorithm, with predictors including reanalysis meteorology, satellite aerosol optical depth (AOD), and Downscaling gridded surface fields from chemical transport models (CTM). The ML model is tested for May 2016, coinciding with the Korea-United States Air Quality Study (KORUS-AQ) intensive field campaign, and for May 2021, to allow incorporation of observations from the Geostationary Environment Monitoring Spectrometer (GEMS). In the tests for May 2016, the correlation coefficients (R) and root mean squared errors (RMSE) relative to withheld observations of daily-averaged pollutants in 10-fold cross-validation are promising: 0.93 (5.5 μg/m<sup>3</sup> ), 0.90 (5.5 ppbv), and 0.95 (4.7 ppbv) for PM2.5, O3, and NO2, respectively. Relative performance is assessed for alternate choices of predictors: (a) 80-km global reanalysis Copernicus Atmosphere Monitoring Service (CAMS) vs. 4-km regional Weather Research and Forecasting model coupled with Chemistry (WRF-Chem); (b) AOD polar-orbiting Moderate Resolution Image Spectroradiometer (MODIS) Multi-Angle Implementation of Atmospheric Correction (MAIAC) vs. geostationary GEMS; and (c) variations in surface observation density. This study is among the very first to incorporate both CTM and GEMS AOD for building surface high resolution multiple air pollution predictions over South Korea. 

## **1. Introduction** 

The number of deaths attributable to indoor and outdoor air pollution currently exceeds 6.7 million per year according to the most recent 

Global Burden of Disease study (Dhimal et al., 2021). Furthermore, ongoing studies are identifying additional health outcomes from exposure to air pollutants, from low-birth-weight newborns (Ebisu and Bell, 2012) to increased risk of dementia in the elderly (Li et al., 2022a). In 

* Corresponding author. 9709 Key West Ave, apt 366, Rockville, MD, USA. _E-mail addresses:_ beiming.tang@noaa.gov, btang6@gmu.edu (B. Tang). 

https://doi.org/10.1016/j.atmosenv.2024.120603 

Received 6 January 2024; Received in revised form 16 May 2024; Accepted 20 May 2024 

Available online 23 May 2024 

1352-2310/© 2024 The Authors. Published by Elsevier Ltd. This is an open access article under the CC BY-NC license (http://creativecommons.org/licenses/bync/4.0/). 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al._ 

addition, air pollution reduces yields of major crops by up to 20% (Avnery et al., 2011; Tong et al., 2007). There remain many important questions on the role of air pollution on human and ecosystem health. We do know that these impacts are disproportionally felt by the most vulnerable, including children, older populations, ethnic minorities, those with underlying health conditions, and under-resourced communities (Colmer et al., 2020). We also know that reducing air pollution exposures requires effective strategies to reduce emissions. However, even with this knowledge of human health and food impacts, progress towards reducing the impacts of air pollution is slow in many parts of the world. 

There is a need for improved estimates of exposure to support air quality management, research and compliance, and analysis of air pollution impacts. Estimates of exposure and impacts rely fundamentally on observations of near-surface air quality from air quality monitoring networks. These networks and number of observation sites are growing in countries around the world, often focused on fine particulate matter (PM2.5), ozone (O3), and nitrogen dioxide (NO2). For example, in the case of South Korea, the number of monitoring locations for PM2.5 (O3) has increased from 164 (311) to 580 (578) between 2016 and 2021. However, the spatial distribution in most countries remains uneven, with more stations in urban regions, and fewer in rural regions (Li et al., 2020). Furthermore, developing countries often have very low ground station density per capita, with only 10% of countries exceeding 3 stations per million people (van Donkelaar et al., 2021). 

One approach to fill gaps in observation networks is by chemical transport models (CTM). CTMs can generate high-resolution spatiotemporally complete coverage for target air pollutants. Another approach is to use satellite observations. Satellite retrieval products also have large domain coverage of column data (top of the atmosphere to the surface of the Earth) for a subset of species. However, both above approaches have limitations. Simulations using CTM require significant expertise and computing infrastructure, and the accuracy of the predictions heavily depends on the meteorology and chemical initial & boundary conditions, emission quality, and reactions included in the model (Geng et al., 2018; Lee et al., 2016; Park et al., 2011). On the other hand, challenges exist to acquire surface ozone and PM2.5 concentrations from satellite column data due to poor correlation. Missing values in satellite observations due to bright surface and clouds and uncertainties in retrieval algorithms are additional problems (Liu et al., 2009). 

Using machine learning (ML) to build spatial-temporal distribution maps for air pollution by combining information from models and measurements is an active area of study. Many previous studies have focused on observation-rich regions (i.e., the U.S., Europe, and China). For example, Zheng et al. (2020) built a ML model in Beijing and tested the model in Shanghai, using a convolutional neural network (CNN) and a random forest (RF) algorithm targeting surface PM2.5 estimates. Chen et al. (2021) used RF ML in the Yangtze River Delta region (including Shanghai, Jiangsu province, and Zhejiang). Wei et al. (2020) found that adding spatiotemporal coordinates as predictors improved the model performance over China. Di et al. (2016) used a CNN algorithm and considered spatial information by adding convolutional layers to build a high-resolution model estimate in the U.S. The four of the above studies all focus on PM2.5. 

In addition, recent studies have adopted ML to predict surface ozone and NO2. Balamurugan et al. (2023) used gradient boosted trees and multilayer perception to model surface ozone and NO2 in Germany. Kang et al. (2021) have tested four ML methods for ozone and NO2 in East Asia: support vector regression (SVR), RF, extreme gradient boost, and light gradient boosting machine (LGBM) and suggested LGBM with leading performance. 

For ozone over East Asia, Gao et al. (2024) used LGBM to estimate hourly 4 km surface mixing ratios incorporating data from the Fengyun-4A satellite. Chen et al. (2023) used a deep learning ML model for estimation at 0.05-degree informed by Himawari remote sensing. 

Wang et al. (2022b) combined deep learning with long short-term memory (LSTM) to estimate daily maximum 8-h (MDA8) ozone at 0.1-degree resolution over four Chinese provinces. 

Numerous models have recently been published for surface NO2 estimation using ML, many incorporating OMI or TROPOMI data. These include LGBM over Vietnam (Ngo et al., 2023), RF over East Asia (Dou et al., 2021; Li et al., 2022b), extra-trees and deep forest model over East Asia (Wei et al., 2022). Ghahremanloo et al. (2021) compared deep CNN, SVR, RF, and multilinear regression over the south-central United States. 

Compared with China and the U.S., South Korea has fewer ground station observations within a smaller domain. Previous studies with a Korean emphasis include Lee et al. (2021) who used both Deep Neural networks (DNN) and RF algorithms to estimate PM2.5 at 4 × 4 km (hourly), with performance for five-fold cross-validation 0.49 (coefficient of determination R<sup>2</sup> ) and 9.17 μg/m<sup>3</sup> (root mean square error RMSE). Lee et al. (2022) also used RF for PM2.5 at 6 × 6 km (hourly) resolution, with model performance on the testing dataset (15% of all data) of 0.69 (R<sup>2</sup> ) and 10.60 μg/m<sup>3</sup> (RMSE). One of the best reported model-observation correlations for Korea was at 6 × 6 km (hourly) resolution, with 0.90 (R<sup>2</sup> ) and 15.77 μg/m<sup>3</sup> (RMSE) on the testing dataset (20% of all data) (Park et al., 2020). However, upon ten-fold cross-validation (split by site), performance decreased to an R<sup>2</sup> of 0.76 and RMSE of 27.41 μg/m<sup>3</sup> . 

The limitations of previously published studies include: 1) focus on observation dense regions; 2) prediction of single species; and 3) neglect potentially useful CTM model products, including long-term public reanalysis datasets. The objective of this study is to further advance the capabilities to fuse measurement and CTM models to produce highresolution near-surface estimates of multiple species (i.e., PM2.5, O3, and NO2), and test ML model dependency on observation density for further application of this ML model in observation scarce regions. 

In this manuscript, we address these limitations using a RF approach over South Korea, for prediction at high spatial resolution for PM2.5, NO2, and O3. South Korea is selected due to its well-established and expanding air quality monitoring network. The availability of Geostationary Environment Monitoring Spectrometer (GEMS) data is also attractive, as it allows an important sensitivity case to compare the influence on ML fusion of lower resolution hourly geostationary aerosol optical depth (AOD) vs. higher resolution but more infrequent AOD from polar-orbiting platforms. Additionally, we have high-resolution chemical transport model outputs, from a contemporary model, extensively evaluated using observations from the Korea and United States Air Quality (KORUS-AQ) field experiment (Park et al., 2021; Tang et al., 2023). Due to their usefulness as a constraint in locations with lower observation density, we incorporate CTM predictors into the analysis and assess sensitivity to high and low spatial resolution CTM fields. Specifically, 4 km Weather Research and Forecasting model coupled with Chemistry (WRF-Chem) modeling done for KORUS-AQ post-analysis is compared the global open-access chemical reanalysis Copernicus Atmospheric Monitoring Service (CAMS) product at 80 km resolution (N et al., 2020). 

The paper is organized as follows. Section two presents the design and structure of the ML model, the algorithm for filling pixels with missing AOD, and the sources of the predictor datasets. Section three presents model performance on four sensitivity test cases. Section four evaluates and discusses model performances related to 1) different CTM inputs, 2) different satellite AOD inputs, and 3) the impact of observational density. 

## **2. Methods** 

## _2.1. Machine learning model workflow_ 

The main ML working flow is shown in Fig. 1 and explained as follows. 

2 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al.                                                                                                                                                                                                                                    Atmospheric Environment 331 (2024) 120603_ 



<!-- Start of picture text -->
‘Avg. to daily, regrid,<br>and gap fill<br>Daily PMs, Oo,<br>Predictor datasets Apply ML NO, estimates at<br>on common 1-km Modal 1km<br>grid<br>—— ; ‘Trained ML Model<br>locations _} extract predictors at<br>observation Performance<br>locations Evaluations<br>avg. to daily<br>/observations paired<br>with predictors<br>Hyperparameter<br>tunning<br>Train and evaluate<br>ML model under 10-<br>fold cross validation<br><!-- End of picture text -->

**Fig. 1.** Modeling workflow. 

Predictor data were extracted to a common domain (33–39<sup>◦</sup> N, 126–130<sup>◦</sup> E), re-gridded to 1 km × 1 km, and averaged to a 24-h basis. Ground truth data were taken from the AirKorea system at 24-h time resolution and split into training and validation subsets using ten-fold cross-validation (Molinaro et al., 2005). Two dimensional predictors were first spatially bilinear interpolated to match ground site latitude and longitude, then temporally interpolated to the observation time. Three ML models were developed in Python (one for each species), using a RF (Breiman, 2001) algorithm (sklearn.ensemble.RandomForestRegressor). Hyperparameter tuning was used to optimize model performance, including n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, and bootstrap. The optimal choices of hyperparameters are presented in Table s1. Once the RF ML models were built, they were applied to create continuous spatiotemporal estimates of PM2.5, O3, and NO2 at 1 km spatial resolution and 24-h temporal resolution. Relative importance of each predictor is reported using Gini importance scores from RandomForest.feature_importances (Breiman, 2001). 

## _2.2. Predictor data sources_ 

We incorporated 1) chemical transport model outputs for PM2.5, O3, and NO2; 2) satellite AOD; 3) meteorological data; 4) elevation data; 5) population data; 5) satellite Normalized Difference Vegetation Index (NDVI); 6) satellite land use cover; and 7) anthropogenic emission data. The detailed information of all input data is summarized in Table s2. 

## _2.2.1. Chemical transport model input_ 

WRF-Chem model output is available over South Korea at 4 km horizontal resolution for the KORUS-AQ period (May 2016), providing hourly surface PM2.5, O3, and NO2 (domain shown in Fig. 2). 

The WRF-Chem results were evaluated within the KORUS-AQ model intercomparison project (Park et al., 2021; Saide et al., 2020) with 

further evaluation reported in Tang et al. (2023). The hourly concentrations in the lowest model layer were averaged to the daily mean and used as input for the ML model. 

As an alternative CTM output, we adopted 0.75<sup>◦</sup> × 0.75<sup>◦</sup> (roughly 80 km × 80 km) CAMS reanalysis data for surface PM2.5, O3, and NO2. The 3-h CAMS reanalysis field was averaged to the daily mean. Monthly mean WRF-Chem and CAMS inputs for PM2.5, O3, and NO2 are plotted in Fig. 3. 

## _2.2.2. Satellite AOD input_ 

For the May 2016 and May 2021 cases, Moderate Resolution Imaging Spectroradiometer (MODIS) Multi-Angle Implementation of Atmospheric Correction (MAIAC) AOD at 550 nm (Chudnovsky et al., 2013; Lyapustin et al., 2011a, 2011b) were adopted, providing 1 km × 1 km resolution AOD. If available, repeat AOD measurements (from TERRA and two from AQUA including overlapping granules) were averaged. 

The high temporal coverage of geospatial products can complement the high spatial resolution but limited temporal resolution of polar remote sensing instruments such as MODIS. On 18 Feb 2020, South Korea launched a geostationary satellite Geostationary Korea MultiPurpose Satellite (GEO-KOMPSAT-2B) with the GEMS on board (Kim et al., 2020). GEMS covers the majority of East and South-East Asia, with six different scan areas. In this study, we used the GEMS level2 AOD (Kim et al., 2020), which has about ten hourly observations per day at nominal spatial resolution of 3.5 km × 8 km. For this work, repeat observations within a day were averaged to a daily mean. GEMS AOD was used for the 2021 case only. 

Both MAIAC and GEMS AOD have missing pixels due to clouds and sun glint. Missing AOD data were filled as follows: First, multiple observations of AOD within a single day (either from MAIAC or GEMS) were averaged to create a daily matrix of 1 km AOD values with missing values. Pixels with three or more valid (AOD) values within a search square of 40 km by 40 km centered on the missing pixel were filled by a 

3 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al.                                                                                                                                                                                                                                    Atmospheric Environment 331 (2024) 120603_ 

organic compounds. 

The date (Julian Day) and location of each ground-level observation were used as predictor variables. Following Wei et al. (2020), location was encoded through seven continuous variables (latitude, longitude, and distances D1 to D5). Distances D1 to D4 were distances to the four domain corners, while D5 was the distance to the domain center. Distances were calculated using the Haversine approximation. 

## _2.3. Observations of PM2.5, O3, NO2_ 

The ML model was trained and validated using observations of PM2.5, O3, and NO2 from South Korean ground-level monitors (htt ps://www.airkorea.or.kr/eng/, mapped in Fig. s2). Details on the number of available observations can be found in Table 1. In summary, in 2016, approximately 150 sites existed for PM2.5 and 300 for gases. This increased to over 550 sites for each of the three target pollutants in 2021. Hourly data were averaged to daily time resolution, and the ML model produced concentration estimates at daily time resolution, corresponding to its training on daily average concentrations. Additional observations from Korea NIER networks (https://www-air.larc.nasa. gov/cgi-bin/ArcView/korusaq?) for PM2.5 were used as withheld measurements to further evaluate the ML model. 

## _2.4. Case selection_ 

To study the influence of different CTM inputs and different satellite AOD inputs, four test cases were run. The cases, listed in Table 1, are & referred to as yyyyab where yyyy refers to the year (2016 or 2021), a 26 refers to the CTM predictor (W for WRF-Chem or C for CAMS reanalysis), Sle and b refers to the AOD predictor (M for MAIAC or G for GEMS). Comparison of 2016WM and 2016CM quantifies the relative influence of the CTM choice. Comparison of 2021CM and 2021CG quantifies the Locations of observarelative influence of the AOD choice. Comparison of 2016CM and 2021CM quantifies the influence of the higher observational site density in 2021, holding the CTM and AOD data source fixed. 

P& 

**Fig. 2. ML modeling domain (33** – **39**<sup>◦</sup> **N, 126** – **130**<sup>◦</sup> **E).** Locations of observation sites with withheld evaluation data from the National Institute of Environmental Research (NIER) network are marked in Seoul Olympic Park, Bulkwang (Incheon), Gwangju, Ulsan. 

weighted average of nearby neighbors using a K-nearest neighbor regression algorithm (Kudraszow and Vieu, 2013) (sklearn.neighbors. KNeighborsRegressor). Those not meeting this criterion were filled using CAMS AOD via bilinear interpolation. The evaluation of CAMS AOD with Aeronet observation are well documented in Kapsomenakis et al. (2022). The flowchart of the above AOD filling technique is shown in Figure s1. 

## _2.2.3. Other predictor data_ 

Meteorological inputs were from the European Centre for MediumRange Weather Forecasts (ECWMF) Reanalysis (ERA)-interim for 2016, and ECWMF Reanalysis v5 (ERA5) for 2021. Eight meteorological variables were used (boundary layer height, dew point at 2 m, evaporation, temperature at 2 m, pressure, precipitation, wind-u at 10 m, wind-v at 10 m). Meteorological predictors were at spatial resolution of 0.1<sup>◦</sup> (roughly 10 km). Elevations were from the Shuttle Radar Topography Mission product at 30 m × 30 m; population data were from the LandScan dataset at 10 km; MODIS land cover data (MCD12Q1) was used at 500 m spatial resolution; categorical land cover data were converted to a continuous variable (urban percentage). MODIS NDVI data at 15-day temporal resolution were interpolated to daily. Emission data at 4 km resolution (KORUS-AQ v5, Woo et al. (2020)) were used for 2016. These covered anthropogenic PM2.5, black carbon, ammonia, nitrogen oxides, and sulfur dioxide. For 2021 emissions, Emission Database for Global Atmospheric Research (EDGAR) Hemispheric Transport of Air Pollution (HTAP) 2018 at 0.1<sup>◦</sup> were used, covering all of the aforementioned species plus anthropogenic non-methane volatile 

## _2.5. Model evaluation_ 

Ten-fold cross-validation (Molinaro et al., 2005) was used to quantify ML model performance. Two different approaches were used for splitting observational data into training and validation subsets. 

We primarily report statistics from a “split by site” approach, where data from 10% of observation sites is withheld from training in each round of cross-validation. This produces errors more representative of expected errors at sites not served by a monitor in the AirKorea network; this is more in line with the expected use of the ML model (i.e., as a way to generate spatially complete gridded air pollution exposure estimates). Observations and model estimates from each round of cross-validation are concatenated, forming paired arrays of observations and model estimates from which statistics (i.e., Pearson correlation coefficient “R” root mean squared error “RMSE”, normalized root mean squared error “NRMSE”, model standard deviation “STD model”, observation standard deviation “STD OBS”, and mean of observation “Mean OBS”) are reported. 

For comparability to other studies, a 2nd cross-validation is used, referred to as “split by sample.” In the split-by-sample approach, 10% of observation samples are removed from training to validation in each round of cross-validation, irrespective of their spatial location. This is much more likely to produce missing days of data at individual sites, and thus the statistics are more representative of the ability of the ML model to gap-fill a missing day or two at a site with data for the days on either side. Withheld observations from separate ground station networks (NIER) not used in training and validating are used to evaluate ML model performance as testing subset. 

4 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al.                                                                                                                                                                                                                                    Atmospheric Environment 331 (2024) 120603_ 



<!-- Start of picture text -->
126°E 128°E 126°E 128°E 126°E 128°E<br>a) c)<br>P = 2 2. “ ”<br>37 a pence 37°N A 7°\ “s =<br>K % 4 KS<br>Rw. Mo Ld<br>9 . i ~“ og<br>{ id 2) <,<br>‘3 AR: —> PEE wal Rieke<br>Be Sine<br>nae<br>33°N 33°N E, N<br>0—10 20 30 740 50 40E._45 50 55 60 65 70 _— O—=E—EEE—0 10 20a30 40<br>, )<br>=<br>N, OJ (|<br>37 “a by} B7°N Sa Pes aie<br>eS Le<br>Bf cy<br>35°N B5°N hs mM Tae abo!<br>33°N B3°N B3°N<br>Om” lms, ——<br>0 10 20 30 40 50 40 45 SO 55 60 65 70 0 10 20 30 40<br>PM, s(ug/m3) Os (ppbv) NO) (ppbv<br><!-- End of picture text -->

**Fig. 3. CTM model inputs used in this study (Mean daily averages from May 1** st **to May 31** st **; at native spatial resolution, 4 km for WRF-Chem, roughly 80 km for CAMS).** a) WRF-Chem PM2.5; b) WRF-Chem O3; c) WRF-Chem NO2; d) CAMS PM2.5; e) CAMS O3; f) CAMS NO2. 

**Table 1** 

Selected predictor datasets for the four cases. 

||Case||||
|---|---|---|---|---|
||2016WM|2016CM|2021CM|2021CG|
|CTM Input|WRF-Chem|CAMS|CAMS|CAMS|
|AOD Input|MAIAC|MAIAC|MAIAC|GEMS|
|PM2.5sites|164|164|580|580|
|O3sites|311|311|578|578|
|NO2sites|318|318|578|578|



_2.6. Workflow for consecutive observation denial_ 

In order to investigate how ML model skill varied with observation density, we built multiple ML models using progressive denial of data. In each iteration, 30 fewer observation sites were used (randomly selected) until the number of observations fell to below 30. 

5 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al._ 

## **3. Results** 

## _3.1. Representative inputs_ 

Fig. 3 shows the WRF-Chem and CAMS ground-level fields of PM2.5, O3, and NO2, which were used as predictors. The higher spatial resolution of WRF-Chem captures features that cannot be resolved in CAMS, including strong urban-rural gradients in all three pollutants, and coastal O3 influence along Korea’s west coast. However, the decrease in PM2.5 from west to east across the Korean peninsula, and aspects of NO2 enhancement and O3 titration in urban centers is also captured at both resolutions. As reanalysis data, CAMS has assimilated AOD, which helps offset the effect of its low resolution on its performance. Overall, 4 km WRF-Chem (R of 0.4–0.65) outperforms CAMS (R of 0.1–0.4) during May 2016. Statistical evaluation against ground station observations (R, RMSE, and STD-model vs. STD-obs) is shown in more detail in the Taylor diagrams (Fig. 5). 

As mentioned in section 2.2, 29 predictor variables are used for each case of the ML model. A selection of these (population, elevation, evaporation, MODIS-MAIAC AOD (May 2016 & May 2021), and GEMS AOD (May 2021)) are mapped in Fig. 4 to help visualize the ML modeling process. 

Relative contribution strengths of predictors are addressed in section 3.4. Fig. 4 contains three monthly averages for AOD (gap filled at daily basis and then averaged to monthly mean). MAIAC 2016 AOD are shown in Fig. 4d; MAIAC 2021 AOD are shown in Fig. 4e. These can be compared to GEMS AOD from the same period (May 2021) in Fig. 4f. The GEMS AOD are notably lower. Cho et al. (2023) reported that GEMS underestimated high AOD values compared with observation from the Aerosol Robotic Network (AERONET). At daily averaging, days with no reported AOD are reduced in GEMS due to ten observation opportunities daily, relative to MAIAC with four per day. In May 2021, GEMS AOD had a median coverage of 67%, while MODIS MAIAC AOD had only 28% for the same time period. A representative plot of GEMS vs. MAIAC AOD on May 7th<sup>,</sup> 2021 before and after gap filling is shown in Figure s3. 

## _3.2. Overview of machine learning fusion result_ 

The 24-h estimates of PM2.5, O3, and NO2 from the ML method had high correlation and low RMSE compared to the original CTM models. The performance statistics in 10-fold cross validation methods (split by site/sample, listed in section 2.5) and for the cases (listed in section 2.4) are graphed in Taylor diagrams in Fig. 5, with detailed statistics in Table s3 and total number of sample in Table s4. For example, ‘2016 WM by sample’ refers to the time period of May-2016, ML with WRF-Chem and MAIAC AOD as input, and evaluated using 10-fold split by sample method. 

The Taylor diagrams show correlation (R), error (centered RMSE), STD-model, and STD-observations. Standard deviation is radius, correlation is angle, and values of RMSE radiate out from the observation point (black star). ML shows marked improvement in the statistical metrics, with only minor differences between the different ML cases. Notice that the statistics reported here in Fig. 5 and Table s3 are for the data withheld in the 10-fold cross validation. Better performance statistics can be generated if training data is also included in the evaluation; however, these are usually not representative of errors in most applications of continuous datasets trained on sparse observations. 

Split-by-sample statistics in general are better than split-by-site results for each sensitive test case, suggesting that the current model does a better job of predicting for stations with missing days than predicting at stations excluded from the current observation networks. This can be seen in the Taylor diagram where split by site cross validation (+symbols) are farther from the observations than split by sample cross validation ( × symbols). The difference is small for PM2.5 case, but larger for O3 and NO2. 

PM2.5 predictions had high correlation and relatively low RMSE. 

Selected statistics for 2016 (2021) under split by sample cross validation are R of 0.93 (0.95) and RMSE of 5.47 (5.43) μg/m<sup>3</sup> . Corresponding 2016 (2021) statistics for the more demanding split by site cross validation are R of 0.91 (0.94) and RMSE of 6.29 (5.68). The statistics for the year 2021 are better than those for 2016 due to enhanced observation density, as discussed in section 4.2. 

For ozone prediction, R was 0.90 for both years, while RMSE improved from around 5.5 ppb to around 4.7 ppb from the year 2016 to the year 2021 in split by sample case. For split by site cases, the 2016 (2021) result was R of 0.73 (0.82) and RMSE of 8.3 (6.1) ppb. 

For NO2 prediction, R values are around 0.95 for both years, while RMSE improved from 4.6 ppb in 2016 to 2.7 ppb in year 2021 for split by sample case. For split by site case, the 2016 (2021) statistics are R of 0.76 (0.81) and RMSE of 9.3 (4.7) ppb. 

## _3.3. Representative outputs_ 

## _3.3.1. Temporal averaged spatial distribution_ 

The spatial distributions of PM2.5, O3, and NO2 generated by ML models in the Korean domain are shown in Fig. 6. 

2016WM and 2021CG cases are selected as representatives for ML model results in the corresponding year. The difference between the four cases and the potential reasons are discussed in section 4.1. Daily 1 km ML results are averaged to generate monthly averages plotted in Fig. 6. We also provide zoom-in results in the capital Seoul region as shown in Fig. 7. The background color is from the ML model, and the color in dots represents daily observations, averaged to monthly mean. 

By comparing model estimation vs. ground station observation, we find they are spatially consistent: PM2.5 and NO2 being high in urban regions and low in rural regions, while ozone is lower in the Capital Seoul compared to the adjacent rural region near Seoul, which is a typical phenomenon due to NO titration. Compared to 2016, air quality in Korea was much improved as shown in Figs. 6 and 7. This improvement in air quality reflects efforts made by both Korea and China to reduce air pollution levels through air quality regulations (Kim et al., 2019; Kim and Lee, 2018; Lu et al., 2020). The modeling framework was also tested for daily maximum 8-h (MDA8) O3 with promising results. Maps and statistical evaluation for an example case ‘2021CM by sample’ of monthly average MDA8 O3 are shown in Figures s4 & s5 and Table s5. 

## _3.3.2. Model performance on temporal trends_ 

To evaluate the ML model’s performance in capturing temporal trends, we graph the 24-h PM2.5 time series at four locations. The locations are Bulkwang (West Coastal), Gwangju (Inland), Seoul Olympic Park (Capital), and Ulsan (Southeast Coastal). These represent different land types (coastal/inland); furthermore, they are key populated cities in Korea. Finally, they are the locations of NIER stations from which we obtained pollutant measurements that were fully withheld from ML model training. 

ML model performance vs. NIER (excluded from training) for PM2.5 are shown in Fig. 8. The withheld NIER time series is shown as black line. In general, ML showed great accuracy in capturing the peak events during May 17th and May 25th. ML model performed extremely well at Seoul Olympic Park and has small bias in other sites. Potential reasons could be the training observation system (AirKorea) has a large percentage of sites located in Seoul. In other words, the observational density at Seoul is much higher compared to others. So the model performance at Seoul is robust to the withholding of individual sites, due to the high observational density. Similar comparisons for ozone and NO2 are shown in Figure s6 and Figure s7. 

ML model performance vs. AirKorea (included in training and 10fold cross validation) for PM2.5 are shown in Fig. 9. AirKorea sites within 10 km of the NIER site are used, and they are compared to the corresponding ML pixels to form the red and grey bands (1-sigma standard deviations). PM2.5 time series of 24-h average concentrations, with shading indicating 1 sigma of standard deviation. 

6 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al.                                                                                                                                                                                                                                    Atmospheric Environment 331 (2024) 120603_ 



<!-- Start of picture text -->
a) 1400<br>1200<br>1072<br>1000<br>200 E ss<br>.§ss<br>$=<br>600 & a~<br>10<br>400<br>200<br>0 10-7<br><i.) 0.7<br>Paes BS ><br>err ee eae ty — bap ae 0.5<br>J as teas & ee S<br>jn“ ere ae 10? g x Oo<br>Se EE i £ § 0.42<br>as : © eae <<br>iS mass CASwes : 1 22b <=<br>Be ae 2 age 03<br>oT eed 5<br>a _ # 10 0.2<br>»<br>07 07<br>06 0.6<br>0.5 0.5<br>8<br>g=<br>3 3<br>0.42 0.49<br>g<br>ZFa<br>= ty<br>0.3 0.3<br>0.2 0.2<br>01 01<br><!-- End of picture text -->

**Fig. 4. Representative predictors regrid to 1 km resolution a) Elevation; b) Evaporation May 2021; c; Population May 2021; d; MAIAC AOD May 2016; e) MAIAC AOD May 2021; f) GEMS AOD May 2021** . All inputs are monthly averages. All AOD panels have gap filling as described in methods. 

7 

_B. Tang et al._ 

_Atmospheric Environment 331 (2024) 120603_ 



<!-- Start of picture text -->
32 0.4 5 Rucn terse] 32 0.4  iitcooysone<br>bos 0. Samo ot mies<br>324 SO, 28 SO,<br>220 o<2 2024 °<a<br>2163212 °CS 1612 °Co<br>2 ky i<br>a8 o «68 °<br>4 Ss 4 3S<br>0 Fook E<br>0 4 8 1216<br>20 24 28 32 0 4 8 1216 2024 28 32<br>16 rad On  isw i sousnrese* 14 rad oe¢  maou<br>=14i$12 Oe° »? oyo.<br>210 e 10 e<br>a 2 8 °<br>2 6 % 6 %<br>é G4 S<br>4<br>°<br>2wo 2 wo<br>0 Bo =<br>0246 810121416 © 0 2 4 6 8 10121416<br>we 2016 08s ‘we 2021085<br>e) 0 02 4 peore. fg o2 | x Seo<br>18 4 % -SX 2ot62016 wecM bychemsample gt----. Me ng % i 20212021 CGCAMS,by sample<br>210 Sec ° 4  \e.<br>g 8) /« aN S ANS<br>&2 6k!by. \\ wo 3 x \ \eln<br>4 o 2 , \e<br>2<br>f<br>0 y \_ Se 1 ; 1 18<br>0 2 4: 6 81012141618: 5 (OO012345678915<br><!-- End of picture text -->

**Fig. 5. Taylor Diagrams. a) 2016 p.m.2.5; b) 2021 p.m.2.5; c) 2016 O3; d) 2021 O3; e) 2016 NO2; f) 2021 NO2.** See text for discussion of the three metrics (correlation, standard deviation, and centered RMSE) graphed in the Taylor diagrams. 

High correlation and low error are seen at all four sites, as one would expect from the overall performance shown in section 3.2 and in the Taylor diagrams. KORUS-AQ contained four phases of pollution transport (Crawford et al., 2021; Peterson et al., 2019). Three of them coincide with our period of analysis (phase 1: dynamic, phase 2: stagnant, and phase 3: long-range transport); they are indicated in Fig. 9, and the ML model reproduces the temporal trend associated with these phases. Standard deviation within the model is lower than in the observations. This is consistent with the Taylor diagrams (Fig. 5a) for 2016WM which shows lower model standard deviation than observation standard deviation. However, Fig. 9 is displaying spatial variability across small urban areas, while the Taylor diagrams are for the overall variability across space and time for all observation sites. 

ML model performances for ozone and NO2 are shown in Figure s6 

and Figure s7, for the split by site cross validation method of case 2016WM. As expected from the Taylor diagrams, performance is still much better than straight CTM model predictions, but not as good as that for PM2.5. Errors are somewhat higher, correlations a bit lower, and the model observation mismatch in standard deviation is larger (insufficient variability in the model, particularly for ozone). For the ozone case, the ML model, in general, captures the range of ozone and captures the peaks on May 5th and May 14th. ML model behaves well in Bulkwang and Seoul region, while worse in Gwangju and Ulsan. For the NO2 case, ML model performance is quite good, the only limitation is the failure to capturing the peak in phase 2 in Ulsan. 

In conclusion, ML models demonstrated considerable skill at capturing temporal trends, which is supported by completely withholding NIER site evaluations and regional evaluations of multiple 

8 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al.                                                                                                                                                                                                                                    Atmospheric Environment 331 (2024) 120603_ 



**Fig. 6. ML model estimates. Top row is 2016; bottom row are 2021. From left to right are PM2.5, O3, and NO2.** a) May 2016 p.m.2.5; b) May 2016 O3; c) May 2016 NO2; d) May 2021 p.m.2.5; e) May 2021 O3; f) May 2021 NO2. Background color is model monthly average of daily estimates; dots are ground observation stations, colored by their monthly average values. 2016 (2021) panels are from the 2016WM (2021CG) case. 

species at different locations in the South Korea domain. 

## _3.4. Input contributions_ 

The relative contributions of the various inputs for the ML models are 

shown in Fig. 10. 

Fig. 10 includes all four configuration cases for three target species (PM2.5, O3, NO2) for the split by sample method. Additional split by site 

input contribution results are presented in Figure s8. 

Temporal information (Julian Day) and spatial information (direct summation of all seven inputs contribution: latitude, longitude, D1, D2, D3, D4, D5) are always among the top 5 contributors for all three species. Julian date is a stronger predictor than location for PM2.5, while for O3 they are balanced, and for NO2 location is a stronger predictor than Julian date. 

CTM input contribution is always high, suggesting that the addition 

9 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al.                                                                                                                                                                                                                                    Atmospheric Environment 331 (2024) 120603_ 



<!-- Start of picture text -->
a) ega ita. Am aeNeva. 50<br>B<br>@ t aoe<br>ag “0: 225 z<br>ans, ” at ~ A\et%4<br>at = at<br>pe a<br>0 , ep as A eX \<br>ep Lil «| SM es<br>M — Bh 4 a ee We ee<br>0<br>(0. aft o Gees ater gree 5 Meco | HAS (‘cue ‘<br>ipa *), aa<br>She: sane Wk sean Nieea. heNebE rd egrr 20) aaYY Ree.\\ oa<br>tits|eSaale  as CSROp) ">eeBooseMORIAcoe\\\\_SRJoe We<br>Lar ii" Ae Se ao ee "<br>oO a<br>2<br>° 3. g rt oe “0<br>cer « FY re<br>0S<br>ca°‘<br>a<br>@ ° b 20<br>WI S me ‘<br>4 1s<br>9 is<br>"o 10<br><!-- End of picture text -->

**Fig. 7. ML model estimates in Seoul. Left are 2016; Right are 2021. From top to bottom are PM2.5, O3, and NO2.** a) May 2016 p.m.2.5; b) May 2016 O3; c) May 2016 NO2; d) May 2021 p.m.2.5; e) May 2021 O3; f) May 2021 NO2. See caption of Fig. 6 for more information. 

of CTM as input is a good choice. Comparing 2016CM with 2016WM cases, the 4-km WRF-Chem contribution is always higher than that of 80-km CAMS in corresponding cases for all three species. This shows higher information content relevant to ML fusion in the high-resolution product. However, even with the CAMS model, the contribution remains ranked highly – except for the case of ozone where it has the same impact as about seven other predictor categories in the 2–10% range. This is supported by Fig. 3, where 80-km CAM resolution cannot capture many fine scale features. 

AOD unsurprisingly serves as one of the major contributors to the PM2.5 ML model (Di et al., 2016; van Donkelaar et al., 2021; Wei et al., 2020). Furthermore, the contribution of AOD to ML predictions of O3 and NO2 are relatively low, with importance less than 5%. Contribution scores for GEMS versus MAIAC are nearly identical (comparing 2021CG with 2021CM cases, which use GEMS and MAIAC AOD, respectively). The impact of AOD dataset selection on the ML model is further 

discussed in section 4.1. 

Wind contribution is the summation of the contributions from the u and v wind velocities used as predictors in ML. The significance of wind is high for PM2.5 (9–21%), moderate for O3 (8–10%) and low for NO2 (3–8%). Specific wind patterns are associated with long-range transport of air pollution in Korea (Crawford et al., 2021; Tang et al., 2023), particularly of PM2.5 as seen during KORUS-AQ in May 2016. We hypothesize the wind fields are being used by the ML model to better represent ventilation effects and long-range transport episodes, although further testing would be needed to better understand how wind fields influence the fused exposure fields. In contrast to PM2.5 (with significant long-range transport influence), NO2 is primarily from Korean domestic sources (Crawford et al., 2021; Tang et al., 2023). Thus, the lower contribution of wind to the ML fusion for NO2 makes sense given its lack of a major long-range transport source. Also, the anthropogenic emission of NOx is a major contributor (around 9–15%) 

10 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al.                                                                                                                                                                                                                                    Atmospheric Environment 331 (2024) 120603_ 



<!-- Start of picture text -->
100<br>ML Model<br>80) —_ NIER OBS Bulkwang<br>25&aE 3 6040} _—ifIXee a—~<br>20 —N —er~ = 2 —_ iw R=0.98RMSE=6.20<br>0<br>100<br>—— ML Model<br>80) — NIER OBS Gwangju<br>SE 60;<br>= 2 40} Va 5<br>{ a — R=0.90<br>20 SS Sw <7 >< RMSE=6.54<br>0<br>100<br>ML Model<br>80) —_ NIER OBS Seoul Olympic Park<br>St 60}<br>& 2 40<br>=<br>20 =NOee— R=0.99RMSE=0.73<br>0<br>100<br>—— ML Model<br>80° — NIEROBS Ulsan<br>“= 60 t<br>=o y/ S -<br>a 40.L /> —~. a= = A yo<br>—~ - . — R=0.94<br>20 — ee ee RMSE=5.82<br>Sor 05/05 05/09 05/13 05/17 0s/21 05/25 05/29<br><!-- End of picture text -->

**Fig. 8.** ML model (split-by-site) predictions (red lines) vs. withheld measurements from Korea NIER observation networks (black lines) at Bulkwang (West coastal), Gwangju (inland), Seoul Olympic Park (Capital), and Ulsan (Southeast coastal) ground stations. 

in NO2 ML models. 

## **4. Discussion** 

_4.1. ML model performance dependency on different model configurations_ 

We further discuss the influence of CTM model selection (4 km vs. 80 km) and AOD data source (MODIS MAIAC vs. GEMS) in this section. Specifically, CTM input is evaluated by comparison of the 2016CM and 2016WM cases. Comparison of the 2021CM and 2021CG cases informs the influence of polar-orbiting satellite MODIS MAIAC AOD vs. geostationary GEMS AOD impact. 

## _4.1.1. Influence of choice of the CTM input_ 

The 4 km WRF vs. 80 km CAMS cases (2016WM and 2016CM) only differ by which CTM fields were entered as a predictor to the ML workflow. Use of the coarse resolution predictor degraded the performance statistics, but only by a very small amount. For example, in the split by site method for PM2.5 prediction and cross validation, RMSE increased from 6.29 to 6.36 μg/m<sup>3</sup> when moving to coarse resolution 

reanalysis CTM data as the predictor. For NO2, RMSE increased from 9.33 to 9.35 ppb. Full model performance statistics are reported in Table s3. 

The monthly average spatial distributions of the two cases are shown in Fig. 11. 

When low resolution CTM data are used as a predictor, the 80 km CAMS pixels become apparent in the model output, as concentration discontinuities at the model pixel edges. This is most apparent in Fig. 11d off the eastern coast of Korea near Ulsan. However, the effect is fairly small, in part due to the small contribution of CTM data to the overall fused product, as explained in section 3.4. (e.g., Fig. 11d). Furthermore, the discontinuities are most apparent over the ocean due to the lack of ground stations and the relative homogeneity of other predictors. 

These results (statistics, spatial distribution, and input contribution) show that using a finer resolution CTM enhances model performance, but the extent of improvement over land is relatively small. 

On the other hand, the drawbacks of incorporating a fine-resolution CTM are: 1) the time and resource requirements to configure, run, and evaluate a CTM for use as a ML model predictor; and 2) the corresponding ML built are limited to the time and domain where the CTM 

11 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al.                                                                                                                                                                                                                                    Atmospheric Environment 331 (2024) 120603_ 



<!-- Start of picture text -->
100; + 1<br>| — Airkorea ops | H<br>go} — Mb Model a) } @ | ai<br>E 60) | |<br>s| } }<br>=|H<br>=40) i<br>20) ; }<br>08/01 05/05 057/09 05/13 05/17 05/21 05/25 05/29<br>100 T<br>— Airkorea oBs } |<br>60) ae ME Moasl 1 } wv | it<br>Ez 60 | |<br>3<br>=40 i H<br>20 / {<br>08/01 05/05 05/09 05/13 05/17 05/21 05/25 05/29<br>100; ; 7<br>| — Airkorea ops H H<br>go) — Mt Model oO } @ | ay<br>= 60 \ i<br>o|H<br>3| H H<br>z 40)<br>20) 1 }<br>08/01 0s/05 05/09 os/i3 0s/i7 05/21 05/25 05/29<br>100 ; T<br>go) — —~ Airkorea Mt Modet ops i | T | aw<br>= 60 | |<br>=Ss H| |<br>=<br>= 40 |{ i<br>: |<br>20<br>08/01 05/05 05/09 05/13 05/17 05/21 05/25 05/29<br><!-- End of picture text -->

**Fig. 9. ML model (split-by-site 2016WM) vs. AirKorea Observations for daily averaged PM2.5 around urban centers** . Bulkwang (13 sites), Gwangju (7 sites), Seoul (18 sites), and Ulsan (4 sites). Shading corresponds to 1 sigma standard deviation. 

12 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al.                                                                                                                                                                                                                                    Atmospheric Environment 331 (2024) 120603_ 



<!-- Start of picture text -->
a)40 mmm 2016WM<br>35 mmm 2016CM<br>mmm<br>mmm = 2021CM2021CG<br>aa<br>&<br>£25<br>8<br>220<br>21s<br>P105<br>5 = ee Oe oe I<br>oS $F SF SL EFS ESEESESEE EES<br>é é & ¢ CS EE EF o FF<br>b)40. mmm 2016WM<br>Bs, @mmmmm 2016CM2021cM<br>ge?= mmm 2021CG<br>5<br>$25<br>220<br>Bis<br>P10s<br>> i mT | in. oi Likbek AE<br>es éges éPL ees $F£ SFCLESFE EZEESE FESES<br>c)40 mmm 2016WM<br>35 jm 2016CM<br>mmm 2021CM<br>ge @mm 2021CG<br>S<br>£25<br>5<br>4<br>£20<br>315<br><€10=<br>° dJekonnd one A oles i i<br>ef SFE SF EFS ESEERSSFEE<br>éé éés ¢ ¢ E EEE&E EEFEESeggs<br><!-- End of picture text -->

**Fig. 10. Input contribution strength a) PM2.5 by sample; b) O3 by sample; c) NO2 by sample.** Selected abbreviations: 2-m dew point (D2m); surface pressure (SP); 2-m temperature (T2m); urban percentage (LUC). Wind is the summation of the contributions from u and v wind velocity predictors. Spatial Info is the summation of contributions from latitude, longitude, and D1, D2, D3, D4, D5 predictors. 

13 

_Atmospheric Environment 331 (2024) 120603_ 



<!-- Start of picture text -->
este ws a ) Dis<br>FO >) ie se<br>Tee i Sta Pita pea<br>See... 2 Tore<br>go Peta Bae.<br>co) Bae pea<br>ae Snea Soa =<br>PM2s (ug/m?) O3 (ppbv) NOz2 (ppbv)<br><!-- End of picture text -->

**Fig. 11. ML models developed based on different CTM inputs.** Top row uses WRF-Chem as CTM input (2016WM). Bottom row uses CAMS as input (2016CM). From left to right are PM2.5, O3, and NO2. 28 out of 29 inputs are identical for both ML models. 

model has been run. The fact that good results are obtained using results from a coarser global model is important. Using results from open access global models makes the ML model applicable globally and at any time of year without the need to run a high-resolution CTM as a prerequisite of ML fusion. 

_4.1.2. Influence of the AOD input_ 

Comparing cases 2021CM with 2021CG, we evaluate the importance of adding GEMS AOD into our ML model. 2021CM uses AOD from a 

polar-orbiting satellite, while 2021CG uses AOD from geostationary satellite (details in section 2.2.2). When preparing AOD input at a daily scale, GEMS AOD has much better coverage (the median coverage for GEMS is 67%, MODIS MAIAC is 28%; GEMS AOD has an increase of roughly 40% median coverage compared to MODIS MAIAC) compared to MODIS MAIAC AOD. 

The model performance of the two cases is shown in Table s3. For the monthly averaged spatial distribution in Fig. 12, no noticeable difference was detected using different AOD as input. 

14 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al._ 



**Fig. 12. ML model estimates holding all predictors equal except AOD input.** Top row has MAIAC AOD input (2021CM). Bottom row has GEMS AOD input (2021CG). From left to right are PM2.5, O3, and NO2. 

This result is at first glance surprising but reflects the fact that inclusion of CTM PM2.5 as a predictor lowers the weights assigned to AOD observations. As shown in Fig. 10, the CTM contribution to PM2.5 prediction is larger than that for AOD in all four cases. Another factor for the lack of difference between the two different AOD inputs is that the higher resolution of MAIAC may offset the higher temporal coverage of GEMS. The advantage of GEMS would be apparent for predictions at higher time resolution (e.g., sub-daily). 

Finally, analysis of GEMS vs. AERONET (Cho et al., 2023) shows an overall correlation (R) of 0.79 and RMSE of 0.227. GEMS AOD showed high bias under clean conditions (AOD443 _<_ 0.47), and low bias under polluted conditions (AOD443 _>_ 0.47). Corresponding statistics for 

MAIAC AOD (evaluated over mainland Korea) were better (Wang et al., 2022a), with R of 0.89–0.91 (both Aqua and Terra) and RMSE of 0.137–0.156. Ranges are due to separate evaluation of MAIAC from MODIS on TERRA vs. AQUA. Using a RF ML model for the bias of GEMS AOD trained on AERONET data, Cho et al. (2023) reported marked improvement in GEMS AOD, with a post correction R of 0.90, an RMSE of 0.16, and a much weaker relationship between AOD and bias. Kim et al. (2024) compared GEMS AOD to AERONET at 680 nm and reported GEMS has a negative bias of − 0.2. Accordingly, advanced retrievals to improve the GEMS AOD error characteristics may be necessary to take full advantage of its coverage improvement. 

15 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al.                                                                                                                                                                                                                                    Atmospheric Environment 331 (2024) 120603_ 

## _4.2. Observation site density effects_ 

Moving from the year 2016–2021, Korea installed four times more ground stations for PM2.5 measurements, and two times more stations for O3 and NO2 measurements as shown in Table 1. This increase in observations provides more data to train the ML model, and leads to improvement in statistics (Table s3 and Fig. 5) when we compare 2021CM with 2016CM. Both cases have the same input for CTM and AOD but a different number of observations. 

We further investigated the impact of the number of observational sites on corresponding ML model performance. ML models were constructed using systematically smaller number of monitoring sites as explained in section 2.6. The decrease in model performance as sites are withheld are shown in Fig. 13. 

Behavior is similar in the 2016 and 2021 cases; however, the 2016 sensitivity is limited to a lower number of observation sites due to the network expansion that occurred from 2016 to 2021. Performance metrics are relatively insensitive to number of sites above about 150 sites (roughly 3 sites per million people), and highly sensitive to observation density at below 150 sites. Detailed statistics in Table s6&s7. 

## _4.3. Comparison to other studies_ 

We compared our ML results with published studies using ML fusion and a focus on South Korea (Lee et al., 2021, 2022; Park et al., 2020). Fig. 14 shows statistics of published studies compared to our ML model. 

Detailed statistics of all related studies can be found in Table s8. The results show our model has the finest spatial resolution and the highest R and lowest RMSE values. However, it should be noted that Fig. 14 represent models estimating PM2.5 for different time periods, using different evaluation schemes, and at different temporal scales (24-h and 1-h). We believe the good statistical performance of the ML 



<!-- Start of picture text -->
16 |<br>Park et al.(2020)RF 6km<br>14<br>12<br>we Lee et al.(2022)RF 6km<br>BE a<br>z3=10 Lee et al.(2021)RF 4km<br>*<br>Lee et al.(2021)DNN 4km<br>8<br>This work,May 2021(1km)<br>6<br>This work,May 2o16(tkr)_—_py<br>0.0 0.2 0.4 0.6 0.8 1.0<br>-<br><!-- End of picture text -->

**Fig. 14. Comparison of ML fusion estimates of PM2.5 over Korea.** ‘Ours 2021’ and ‘Ours 2016’ are from this work; ‘Lee (2021) DNN’ is reported by Lee et al. (2021) using Deep Neural Network with five-fold cross validation to produce hourly PM2.5; ‘Lee (2021) RF’ is reported by Lee et al. (2021) using RF with five-fold cross validation to produce hourly PM2.5; ‘Lee (2022) RF’ is reported by Lee et al. (2022) using RF on a 15% testing dataset for hourly PM2.5 during daytime; ‘Park (2020) RF’ is reported by Park et al. (2020) using RF with 10-fold cross validation for hourly PM2.5 during daytime. Case shown for this work is (2016WM and 2016CM) with split by sample 10-fold cross validation. 



<!-- Start of picture text -->
0 2 site4 Pp per million6 p peoplePp 8 10<br>0.950 10<br>0.925<br>—— R2021 /}»<br>0.900<br>0.875, M —— R 2016 =<br>¢ 8sE<br>! io))<br>cc 0.850 a<br>0.825 \ —e— RMSE 2021 |, &<br>\ —e— RMSE 2016 =<br>\\<br>0.800 wr 6<br>0.775<br>0.750 5<br>0 100 200 300 400 500 600<br>number of sites<br><!-- End of picture text -->

**Fig. 13. Model statistics (R & RMSE) and observation density.** Cases shown are 2021CM and 2016 CM. split by site 10-fold cross validation method. 

16 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al.                                                                                                                                                                                                                                    Atmospheric Environment 331 (2024) 120603_ 

configuration we developed is primarily due to the configuration of predictors used (spatiotemporal coordinates, meteorology, CTM, etc.). 

map_MODEL_20160501_R0_thru20160610-WRFChemUCLA.nc” for surface predictions. Machine learning output daily, 1 km PM2.5, O3, NO2 data are accessible upon request with author. 

## **5. Conclusion** 

An accurate, computationally efficient, and robust estimate for surface air pollutants has many important uses. In this study, we presented a ML data fusion approach to provide multi-species (PM2.5, O3, NO2) near surface estimates of concentrations at high resolution (1km × 1 km), by incorporating routine ground observations, established CTM fields, and satellite remote sensing data. 

Demonstrated for May 2016 and May 2021, the ML fusion produces spatially continuous daily exposure estimates at 24-h time resolution. For the year 2016, we found R = 0.93/0.90/0.95 and RMSE = 5.47 μg/ m<sup>3</sup> /5.46 ppb/4.65 ppb for PM2.5/O3/NO2 in ten-fold cross-validation split by sample method; and R = 0.91/0.73/0.76 and RMSE = 6.29 μg/ m<sup>3</sup> /8.27 ppb/9.33 ppb for PM2.5/O3/NO2 for split by site method. For year 2021, we found R = 0.95/0.90/0.94 and RMSE = 5.44 μg/m<sup>3</sup> /4.69 ppb/2.73 ppb for PM2.5/O3/NO2 in ten-fold cross-validation split by sample method; and R = 0.94/0.82/0.81 and RMSE = 5.66 μg/m<sup>3</sup> /6.10 ppb/4.75 ppb for PM2.5/O3/NO2 for split by site method. 

The most influential predictors were spatiotemporal parameters (i.e., Julian data and location within the domain). Other important predictors varied by pollutant and case, but included CTM fields, wind speed, primary emissions, and other meteorological variables (i.e., evaporation, precipitation, boundary layer height). Ground station observational density was shown to have a major effect on ML model performance. The expansion of the monitoring network from 2016 to 2021 improved ML model skill significantly for all three pollutants. Experiments with synthetic deletion of PM2.5 monitor data indicate a threshold at 150 monitors (roughly 3 monitors per million persons). Skill fell off sharply at observational density below this value, and continuously improved for larger numbers of monitors. 

Sensitivity of ML fusion skill to CTM resolution and AOD data source was also assessed. The use of high-resolution (4 km) WRF-Chem fields had a small performance edge relative to 80 km CAMS reanalysis. However, the wide availability of CAMS reanalysis coupled with the small difference in performance is promising for use of chemical reanalysis data in applications where higher resolution CTM inputs are not available. ML model performance was also compared with two different AOD data sources, one geostationary with fewer missing pixels per day (GEMS), and one polar orbiting with more missing pixels per day (MODIS MAIAC). We found that the ML model incorporating GEMS satellite AOD performed slightly better in terms of statistics (R, RMSE). 

Our ML model is built on widely available, well-documented, and mature input data sources. Thus, it can be applied anywhere in the world. The availability of surface-based observations is a limiting factor impacting model performance. However, we have applied this model to data poor regions (e.g., Vietnam) and achieved useful results (Christiansen et al., 2022). We are currently evaluating ML model performance by incorporating low-cost sensor measurements, which are becoming more widely available in many countries. 

In future studies we plan to use satellite radiances directly as input. Finally, this current study used the same set of inputs to build ML models for the various pollutants (PM2.5, O3 and NO2). While the current input selection proved to yield satisfying results for PM2.5 and NO2 cases, the ozone input selection could be refined. We plan to add a feature selection procedure to optimize inputs for the ozone model. 

## **Data accessibility statement** 

Hourly surface observations from the AirKorea network can be downloaded through (https://www.airkorea.or.kr/eng/hourlyTrends? pMENU_NO=151). WRF-Chem modeled data can be accessed through (https://www-air.larc.nasa.gov/cgi-bin/ArcView/korusaq?MODEL=1) under group “PARK.ROKJIN” with filename “korusaq_surface- 

## **Funding sources** 

This research has been supported by the NASA SERVIR grant 80NSSC23K0244, NASA KORea and United-States Air Quality Study grant NNX15AU17G, NASA Health and Air Quality Applied Science Team grant NNX16AQ19G, and NASA ACMAP grant 80NSSC19K0946. 

## **CRediT authorship contribution statement** 

**Beiming Tang:** Writing – original draft, Visualization, Validation, Methodology, Investigation, Formal analysis, Data curation, Conceptualization. **Charles O. Stanier:** Writing – review & editing, Writing – original draft, Supervision, Resources, Project administration, Investigation, Funding acquisition, Formal analysis, Conceptualization. **Gregory R. Carmichael:** Writing – review & editing, Writing – original draft, Supervision, Resources, Project administration, Investigation, Funding acquisition, Conceptualization. **Meng Gao:** Writing – review & editing, Methodology, Investigation, Formal analysis, Conceptualization. 

## **Declaration of competing interest** 

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. 

## **Data availability** 

Data will be made available on request. 

## **Acknowledgement** 

The authors acknowledge AirKorea network system to provide surface observations of PM2.5, O3, NO2. (https://www.airkorea.or.kr/en g/hourlyTrends?pMENU_NO1/4151). Surface PM2.5 data from NIER stations during KORUS-AQ from the campaign data archieve (https ://www-air.larc.nasa.gov/cgi-bin/ArcView/korusaq?). The authors thank Dr. Daniel Tong for providing review of draft of paper. 

## **Appendix A. Supplementary data** 

Supplementary data to this article can be found online at https://doi. org/10.1016/j.atmosenv.2024.120603. 

## **References** 

- Avnery, S., Mauzerall, D.L., Liu, J.F., Horowitz, L.W., 2011. Global crop yield reductions due to surface ozone exposure: 1. Year 2000 crop production losses and economic damage. Atmos. Environ. 45, 2284–2296. 

- Balamurugan, V., Chen, J., Wenzel, A., Keutsch, F.N., 2023. In: Spatio-temporal Modeling of Air Pollutant Concentrations in Germany Using Machine Learning. EGUsphere, pp. 1–28, 2023. 

- Breiman, L., 2001. Random forests. Mach. Learn. 45, 5–32. 

- Chen, B., Wang, Y., Huang, J., Zhao, L., Chen, R., Song, Z., Hu, J., 2023. Estimation of near-surface ozone concentration and analysis of main weather situation in China based on machine learning model and Himawari-8 TOAR data. Sci. Total Environ. 864, 160928. 

- Chen, B.J., Lin, Y., Deng, J.S., Li, Z.Y., Dong, L., Huang, Y.B., Wang, K., 2021. Spatiotemporal dynamics and exposure analysis of daily PM2.5 using a remote sensing-based machine learning model and multi-time meteorological parameters. Atmos. Pollut. Res. 12, 23–31. 

- Cho, Y., Kim, J., Go, S., Kim, M., Lee, S., Kim, M., Chong, H., Lee, W.J., Lee, D.W., Torres, O., Park, S.S., 2023. First atmospheric aerosol monitoring results from geostationary environment monitoring spectrometer (GEMS) over Asia. Atmos. Meas. Tech. Discuss 2023, 1–29. 

- Christiansen, M., Carmichael, G.R., Tang, B.M., Stanier, C.O., Blount, R., 2022. Multipollutant high resolution exposure assessment in Vietnam, 12-16 December 2022. In: 

17 

_Atmospheric Environment 331 (2024) 120603_ 

_B. Tang et al._ 

Support of Tuberculosis Research. AGU Fall Meeting 2022, Held in Chicago, IL, p. A43A, 06. 

- Chudnovsky, A., Lyapustin, A., Wang, Y., Schwartz, J., Koutrakis, P., 2013. Analyses of high resolution aerosol data from MODIS satellite: a MAIAC retrieval, southern New England, US. Remote Sensing and Geoinf. Environ. (RSCY2013), 8795. 

- Colmer, J., Hardman, I., Shimshack, J., Voorheis, J., 2020. Disparities in PM2.5 air pollution in the United States. Science 369, 575–578. 

- Crawford, J.H., Ahn, J.Y., Al-Saadi, J., Chang, L., Emmons, L.K., Kim, J., Lee, G., Park, J. H., Park, R.J., Woo, J.H., Song, C.K., Hong, J.H., Hong, Y.D., Lefer, B.L., Lee, M., Lee, T., Kim, S., Min, K.E., Yum, S.S., Shin, H.J., Kim, Y.W., Choi, J.S., Park, J.S., Szykman, J.J., Long, R.W., Jordan, C.E., Simpson, I.J., Fried, A., Dibb, J.E., Cho, S., Kim, Y.P., 2021. The Korea-United States air quality (KORUS-AQ) field study. Elementa-Sci Anthrop 9. 

- Dhimal, M., Chirico, F., Bista, B., Sharma, S., Chalise, B., Dhimal, M.L., Ilesanmi, O.S., Trucillo, P., Sofia, D., 2021. Impact of air pollution on global burden of Disease in 2019. Processes 9. 

- Di, Q., Kloog, I., Koutrakis, P., Lyapustin, A., Wang, Y.J., Schwartz, J., 2016. Assessing PM2.5 exposures with high spatiotemporal resolution across the continental United States. Environ. Sci. Technol. 50, 4712–4721. 

- Dou, X., Liao, C., Wang, H., Huang, Y., Tu, Y., Huang, X., Peng, Y., Zhu, B., Tan, J., Deng, Z., Wu, N., Sun, T., Ke, P., Liu, Z., 2021. Estimates of daily ground-level NO2 concentrations in China based on Random Forest model integrated K-means. Adv. Appl. Energy 2, 100017. 

- Ebisu, K., Bell, M.L., 2012. Airborne PM2.5 chemical components and low birth weight in the northeastern and mid-atlantic regions of the United States. Environ. Health Perspect. 120, 1746–1752. 

- Gao, L., Zhang, H., Yang, F., Tan, W., Wu, R., Song, Y., 2024. First estimation of hourly full-coverage ground-level ozone from Fengyun-4A satellite using machine learning. Environ. Res. Lett. 19, 024040. 

- Geng, G.N., Murray, N.L., Tong, D., Fu, J.S., Hu, X.F., Lee, P., Meng, X., Chang, H.H., Liu, Y., 2018. Satellite-based daily PM2.5 estimates during fire seasons in Colorado. J. Geophys. Res. Atmos. 123, 8159–8171. 

- Ghahremanloo, M., Lops, Y., Choi, Y., Yeganeh, B., 2021. Deep learning estimation of daily ground-level NO2 concentrations from remote sensing data. J. Geophys. Res. Atmos. 126, e2021JD034925. 

- Kang, Y., Choi, H., Im, J., Park, S., Shin, M., Song, C.-K., Kim, S., 2021. Estimation of surface-level NO2 and O3 concentrations using TROPOMI data and machine learning over East Asia. Environ. Pollut. 288, 117711. 

- Kapsomenakis, J., Zerefos, C., Langerock, B., Errera, Q., Basart, S., Cuevas, E., Bennouna, Y., Thouret, V., Arola, A., Pitkanen, M.R.A., Blechschmidt, A.-M., Richter, A., Eskes, H.J., Tsikerdekis, T., Benedictow, A., Schulz, M., Bouarar, I., Warneke, T., 2022. Validation report of the CAMS global Reanalysis of aerosols and reactive gases, years 2003-2021. In: Copernicus Atmosphere Monitoring Service (CAMS) Report. 

- Kim, H., Chen, X., Wang, J., Lu, Z.Z.M., Carmichael, G., Park, S.K.J., 2024. Aerosol layer height (ALH) retrievals from oxygen absorption bands: intercomparison and validation among different satellite platforms, GEMS, EPIC, and TROPOMI. Atmos. Meas. Tech. (submitted). 

- Kim, H., Kim, H., Lee, J.T., 2019. Effect of air pollutant emission reduction policies on hospital visits for asthma in Seoul, Korea; Quasi-experimental study. Environ. Int. 132. 

- Kim, J., Jeong, U., Ahn, M.H., Kim, J.H., Park, R.J., Lee, H., Song, C.H., Choi, Y.S., Lee, K. H., Yoo, J.M., Jeong, M.J., Park, S.K., Lee, K.M., Song, C.K., Kim, S.W., Kim, Y.J., Kim, S.W., Kim, M., Go, S., Liu, X., Chance, K., Chan Miller, C., Al-Saadi, J., Veihelmann, B., Bhartia, P.K., Torres, O., Abad, G.G., Haffner, D.P., Ko, D.H., Lee, S. H., Woo, J.H., Chong, H., Park, S.S., Nicks, D., Choi, W.J., Moon, K.J., Cho, A., Yoon, J., Kim, S.K., Hong, H., Lee, K., Lee, H., Lee, S., Choi, M., Veefkind, P., Levelt, P.F., Edwards, D.P., Kang, M., Eo, M., Bak, J., Baek, K., Kwon, H.A., Yang, J., Park, J., Han, K.M., Kim, B.R., Shin, H.W., Choi, H., Lee, E., Chong, J., Cha, Y., Koo, J.H., Irie, H., Hayashida, S., Kasai, Y., Kanaya, Y., Liu, C., Lin, J., Crawford, J. H., Carmichael, G.R., Newchurch, M.J., Lefer, B.L., Herman, J.R., Swap, R.J., Lau, A. K.H., Kurosu, T.P., Jaross, G., Ahlers, B., Dobber, M., McElroy, C.T., Choi, Y., 2020. New Era of air quality monitoring from space: geostationary environment monitoring spectrometer (GEMS). Bull. Am. Meteorol. Soc. 101, E1–E22. 

- Kim, Y.P., Lee, G., 2018. Trend of air quality in Seoul: policy and science. Aerosol Air Qual. Res. 18, 2141–2156. 

- Kudraszow, N.L., Vieu, P., 2013. Uniform consistency of NN regressors for functional 

   - variables. Stat. Probab. Lett. 83, 1863–1870. 

- Lee, C., Lee, K., Kim, S., Yu, J., Jeong, S., Yeom, J., 2021. Hourly ground-level PM2.5 estimation using geostationary satellite and reanalysis data via deep learning. Remote Sens-Basel 13. 

- Lee, S., Park, S., Lee, M.I., Kim, G., Im, J., Song, C.K., 2022. Air quality forecasts improved by combining data assimilation and machine learning with satellite AOD. Geophys. Res. Lett. 49. 

- Lee, S., Song, C.H., Park, R.S., Park, M.E., Han, K.M., Kim, J., Choi, M., Ghim, Y.S., Woo, J.H., 2016. GIST-PM-Asia v1: development of a numerical system to improve particulate matter forecasts in South Korea using geostationary satellite-retrieved aerosol optical data over Northeast Asia. Geosci. Model Dev. (GMD) 9, 17–39. 

- Li, J., Wang, Y.F., Steenland, K., Liu, P.F., van Donkelaar, A., Martin, R.V., Chang, H.H., Caudle, W.M., Schwartz, J., Koutrakis, P., Shi, L.H., 2022a. Long-term effects of PM2.5 components on incident dementia in the northeastern United States. Innovation-Amsterdam 3. 

- Li, M., Wu, Y., Bao, Y., Liu, B., Petropoulos, G.P., 2022b. Near-surface NO2 concentration estimation by random forest modeling and sentinel-5P and ancillary data. Remote Sens-Basel 14, 3612. 

- Li, T.W., Shen, H.F., Zeng, C., Yuan, Q.Q., 2020. A validation approach considering the uneven distribution of ground stations for satellite-based PM2.5 estimation. LEEE J. Selected Topics in Appl. Earth Observations and Remote Sens. 13, 1312–1321. 

- Liu, Y., Paciorek, C.J., Koutrakis, P., 2009. Estimating regional spatial and temporal variability of PM2.5 concentrations using satellite data, meteorology, and land use information. Environ. Health Perspect. 117, 886–892. 

- Lu, X., Zhang, S.J., Xing, J., Wang, Y.J., Chen, W.H., Ding, D., Wu, Y., Wang, S.X., Duan, L., Hao, J.M., 2020. Progress of air pollution control in China and its challenges and opportunities in the ecological civilization era. Eng. Plast. 6, 1423–1431. 

- Lyapustin, A., Martonchik, J., Wang, Y.J., Laszlo, I., Korkin, S., 2011a. Multiangle implementation of atmospheric correction (MAIAC): 1. Radiative transfer basis and look-up tables. J. Geophys. Res. Atmos. 116. 

- Lyapustin, A., Wang, Y., Laszlo, I., Kahn, R., Korkin, S., Remer, L., Levy, R., Reid, J.S., 2011b. Multiangle implementation of atmospheric correction (MAIAC): 2. Aerosol algorithm. J. Geophys. Res. Atmos. 116. 

- Molinaro, A.M., Simon, R., Pfeiffer, R.M., 2005. Prediction error estimation: a comparison of resampling methods. Bioinformatics 21, 3301–3307. 

- N, B., D, W., S, K., M, J., Q, J., F, P., R, L., S, C., B, G., S, N., B, I., 2020. Documentation of CAMS climate forcing products, version 2. ECMWF COPERNICUS REPORT. December, 2020. 

- Ngo, T.X., Phan, H., Nguyen, T.T.N., 2023. Development of ground-level NO2 models in Vietnam using machine learning and satellite observations with ancillary data. Front. Environ. Sci. 11. 

- Park, R.J., Oak, Y.J., Emmons, L.K., Kim, C.H., Pfister, G.G., Carmichael, G.R., Saide, P. E., Cho, S.Y., Kim, S., Woo, J.H., Crawford, J.H., Gaubert, B., Lee, H.J., Park, S.Y., Jo, Y.J., Gao, M., Tang, B.M., Stanier, C.O., Shin, S.S., Park, H.Y., Bae, C., Kim, E., 2021. Multi-model intercomparisons of air quality simulations for the KORUS-AQ campaign. Elementa-Sci Anthrop 9. 

- Park, R.S., Song, C.H., Han, K.M., Park, M.E., Lee, S.S., Kim, S.B., Shimizu, A., 2011. A study on the aerosol optical properties over East Asia using a combination of CMAQ-simulated aerosol optical properties and remote-sensing data via a data assimilation technique. Atmos. Chem. Phys. 11, 12275–12296. 

- Park, S., Lee, J., Im, J., Song, C.K., Choi, M., Kim, J., Lee, S., Park, R., Kim, S.M., Yoon, J., Lee, D.W., Quackenbush, L.J., 2020. Estimation of spatially continuous daytime particulate matter concentrations under all sky conditions through the synergistic use of satellite-based AOD and numerical models. Sci. Total Environ. 713. 

- Peterson, D.A., Hyer, E.J., Han, S.O., Crawford, J.H., Park, R.J., Holz, R., Kuehn, R.E., Eloranta, E., Knote, C., Jordan, C.E., Lefer, B.L., 2019. Meteorology influencing springtime air quality, pollution transport, and visibility in Korea. Elementa-Sci Anthrop 7. 

- Saide, P.E., Gao, M., Lu, Z.F., Goldberg, D., Streets, D.G., Woo, J.H., Beyersdorf, A., Corr, C.A., Thornhill, K.L., Anderson, B., Hair, J.W., Nehrir, A.R., Diskin, G.S., Jimenez, J.L., Nault, B.A., Campuzano-Jost, P., Dibb, J., Heim, E., Lamb, K.D., Schwarz, J.P., Perring, A.E., Kim, J., Choi, M., Holben, B., Pfister, G., Hodzic, A., Carmichael, G.R., Emmons, L., Crawford, J.H., 2020. Understanding and improving model representation of aerosol optical properties for a Chinese haze event measured during KORUS-AQ. Atmos. Chem. Phys. 20, 6455–6478. 

- Tang, B.M., Saide, P.E., Gao, M., Carmichael, G.R., Stanier, C.O., 2023. WRF-Chem quantification of transport events and emissions sensitivity in Korea during KORUSAQ. Elementa-Sci Anthrop 11. 

- Tong, D., Mathur, R., Schere, K., Kang, D., Yu, S., 2007. The use of air quality forecasts to assess impacts of air pollution on crops: Methodology and case study. Atmos. Environ. 41, 8772–8784. 

- van Donkelaar, A., Hammer, M.S., Bindle, L., Brauer, M., Brook, J.R., Garay, M.J., Hsu, N.C., Kalashnikova, O.V., Kahn, R.A., Lee, C., Levy, R.C., Lyapustin, A., Sayer, A.M., Martin, R.V., 2021. Monthly global estimates of fine particulate matter and their uncertainty. Environ. Sci. Technol. 55, 15287–15300. 

- Wang, P., Tang, Q.X., Zhu, Y.X., Zheng, K., Liang, T.Q., Yu, Q.Z., He, Y.Q., 2022a. Validation and analysis of MAIAC AOD aerosol products in East Asia from 2011 to 2020. Remote Sens-Basel 14. 

- Wang, S., Mu, X., Jiang, P., Huo, Y., Zhu, L., Zhu, Z., Wu, Y., 2022b. New deep learning model to estimate ozone concentrations found worrying exposure level over eastern China. Int. J. Environ. Res. Publ. Health 19. 

- Wei, J., Li, Z.Q., Cribb, M., Huang, W., Xue, W.H., Sun, L., Guo, J.P., Peng, Y.R., Li, J., Lyapustin, A., Liu, L., Wu, H., Song, Y.M., 2020. Improved 1 km resolution PM2.5 estimates across China using enhanced space-time extremely randomized trees. Atmos. Chem. Phys. 20, 3273–3289. 

- Wei, J., Liu, S., Li, Z., Liu, C., Qin, K., Liu, X., Pinker, R.T., Dickerson, R.R., Lin, J., Boersma, K.F., Sun, L., Li, R., Xue, W., Cui, Y., Zhang, C., Wang, J., 2022. Groundlevel NO2 surveillance from space across China for high resolution using interpretable spatiotemporally weighted artificial intelligence. Environ. Sci. Technol. 56, 9988–9998. 

- Woo, J.H., Kim, Y., Kim, H.K., Choi, K.C., Eum, J.H., Lee, J.B., Lim, J.H., Kim, J., Seong, M., 2020. Development of the CREATE inventory in support of integrated climate and air quality modeling for Asia. Sustain.-Basel 12. 

- Zheng, T.S., Bergin, M.H., Hu, S.J., Miller, J., Carlson, D.E., 2020. Estimating groundlevel PM2.5 using micro-satellite images by a convolutional neural network and random forest approach. Atmos. Environ. 230. 

18 

