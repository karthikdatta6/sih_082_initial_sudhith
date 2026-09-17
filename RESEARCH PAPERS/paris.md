

<!-- Start of picture text -->
PLOsY. One<br><!-- End of picture text -->

##### RESEARCH ARTICLE 

# **Forecasting urban air quality in Paris using ensemble machine learning: A scalable framework for environmental management** 

**Somia A. Asklany 1** * **, Doaa Mohammed1, Ismail K. Youssef2, Majed Nawaz1, Wajdan Al Malwi**<sup>**3**</sup> 

**1** Department of Computer Science, College of Science, Northern Border University, Arar, Saudi Arabia, **2** Department of Mathematics, Faculty of Science, Islamic University of Madinah, Madinah, Saudi Arabia, **3** Department of Informatics and Computer Systems, College of Computer Science, King Khalid University, Abha, Saudi Arabia 



<!-- Start of picture text -->
®<br>checkfo,<br><!-- End of picture text -->



<!-- Start of picture text -->
6<br><!-- End of picture text -->

##### OPEN ACCESS 

**Citation:** Asklany SA, Mohammed D, Youssef IK, Nawaz M, Al Malwi W (2025) Forecasting urban air quality in Paris using ensemble machine learning: A scalable framework for environmental management. PLoS One 20(11): e0336897. <u>https://doi.org/10.1371/journal. pone.0336897</u> 

**Editor:** Majid Niazkar, Euro-Mediterranean Center for Climate Change: Fondazione Centro Euro-Mediterraneo sui Cambiamenti Climatici, ITALY 

* <u>somia.asklany@nbu.edu.sa</u> 

## **Abstract** 

Urban air pollution poses a significant threat to public health and urban sustainability in megacities like Paris. We cast forecasting as a short-term, next-hour prediction task for PM2.5, NO, and CO, using hourly meteorology and recent pollutant history as inputs. We develop a data-driven framework based on hyperparameter-tuned ensembles (Random Forest, Gradient Boosting, and a Stacked Ensemble) and benchmark against a Long Short-Term Memory (LSTM) model, alongside persistence baselines. All evaluation metrics (RMSE/MAE) are reported in physical units (µg/m³) with R² unitless. Results show that tree ensembles deliver the lowest errors for PM2.5 and CO, while LSTM is competitive for NO; stacking offers gains when base-model errors are complementary but does not universally dominate. The framework is designed for real-time deployment and integration into smart city pipelines, supporting proactive air quality management. By providing accurate, unit-consistent short-term forecasts, this study informs urban planning, risk mitigation, and public-health protection. 

**Received:** July 25, 2025 

**Accepted:** October 28, 2025 

**Published:** November 20, 2025 

**Copyright:** © 2025 Asklany et al. This is an open access article distributed under the terms of the <u>Creative Commons Attribution License,</u> which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited. 

**Data availability statement:** Data for this study are publicly available from the Zenodo repository (https://doi.org/10.5281/ <u>zenodo.17167030). Code for this study is</u> 

### **1. Introduction** 

Air pollution is a significant threat to global public health and urban sustainability, contributing to an estimated 7 million premature deaths worldwide each year, according to the World Health Organization. In densely populated metropolitan areas like Paris, the impacts of air pollution are particularly acute. The city’s intricate web of vehicular traffic, industrial zones, residential heating systems, and seasonal meteorological patterns interacts in complex ways to exacerbate the levels of harmful pollutants. These include fine particulate matter (PM2.5 and PM10), nitrogen oxides (NO and NO₂), and carbon monoxide (CO), all of which are associated with a range of health outcomes, from chronic respiratory illnesses and cardiovascular disease to cognitive decline and premature mortality [1]. 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

1 / 17 



publicly available from the GitHub repository (https://github.com/SomiaAsklany/ <u>using-ensemble-machine-learning).</u> 

**Funding:** The authors extend their appreciation to the Deanship of Scientific Research at Northern Border University, Arar, Saudi Arabia, for funding this research work through the Project No. NBU-FFR- 2025-2932-12 The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript. 

**Competing interests:** The authors have declared that no competing interests exist. 

**Abbreviations:** PM, Particulate Matter; **PM2.5** , Particulate matter ≤2.5 µm; **PM10** , Particulate matter ≤10 µm; **NO** , Nitric Oxide; **NO2** , Nitrogen Dioxide; **CO** , Carbon Monoxide; **RF** , Random Forest; **GBM** , Gradient Boosting Machine; **LSTM** , Long Short-Term Memory; **RMSE** , Root Mean Square Error; **MAE** , Mean Absolute Error; **R**<sup>**2**</sup> , Coefficient of Determination; **CI** , Confidence Interval; **CV** , Cross-Validation; **OOF** , Out-of-Fold; **NOAA** , National Oceanic and Atmospheric Administration; **OpenAQ** , Open Air Quality platform; **PACE** , Preflight Analysis and Conversion Engine. 

Paris has taken several legislative and infrastructural steps to mitigate these challenges, such as implementing the Paris Climate Plan, enforcing low- emission zones, and restricting vehicular access in certain districts during pollution peaks [2]. While these measures are impactful, their effectiveness hinges on access to accurate, high-resolution air quality forecasts that enable city planners and public health authorities to act proactively. Yet, traditional air quality models, including deterministic chemical transport models and regression-based statistical approaches, often fall short in dynamically capturing the nonlinear interactions between environmental, anthropogenic, and meteorological variables. In response, data-driven methods such as machine learning (ML) have emerged as promising alternatives [3,4]. ML models, particularly ensemble learning techniques, can accommodate high-dimensional, complex, and noisy datasets without rigid assumptions. However, deploying such models for urban air quality forecasting requires careful calibration, feature engineering, and validation to ensure both interpretability and operational reliability. Moreover, despite their technical merit, ML models must be contextualized within the broader environmental and policy landscape to ensure scientific rigor and real-world applicability [5]. 

This study addresses this intersection by presenting a scientifically grounded, hyperparameter-tuned ensemble learning framework for air pollution forecasting, using Paris as a case study. We combine Random Forest, Gradient Boosting, and a Stacked Ensemble model (with LightGBM as the meta-learner), trained on hourly air quality and meteorological data from 2023 [6,7]. The framework is benchmarked against a Long Short-Term Memory (LSTM) deep learning model to assess accuracy across pollutants with varying temporal dynamics. Beyond technical performance, we explore the implications of model deployment for real-time environmental intelligence, urban sustainability planning, and public health policy [8,9]. By embedding predictive models within the complex realities of a megacity, this research aims to provide a replicable and scalable solution for cities seeking to balance growth, resilience, and environmental health. 

In this study, we formulate air quality forecasting in Paris as a short-term, nexthour prediction problem, where the task is to predict concentrations of three key pollutants (PM2.5, NO, and CO) using hourly meteorological variables (temperature, wind speed, sea-level pressure, and visibility) and recent pollutant observations as input features. The workflow of this paper is summarized in <u>Fig 1.</u> 

### **2. Related work** 

Air pollution forecasting has been an active area of research due to its significant impact on public health, environmental sustainability, and urban planning. Traditional approaches, including statistical regression and atmospheric dispersion models, have been widely used but often struggle to capture the complex and dynamic nature of air pollution. With advancements in machine learning (ML) and artificial intelligence (AI), researchers have explored various data-driven models to improve forecasting accuracy. 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

2 / 17 



<!-- Start of picture text -->
PLosy. One<br><!-- End of picture text -->



<!-- Start of picture text -->
Data Precocessing<br>DATA SET Handling Missing Values<br>Outlier Detection and<br>Removal<br>Normalization an Scaling<br>a<br>Feature Selection<br>—" oad o— >|)<br>c o-oo o—<br>RandomModelForest Boosting Model StackedModelEnsemble<br>Conclusion<br><!-- End of picture text -->

**Fig 1. The workflow of the submitted proposal.** 

<u>https://doi.org/10.1371/journal.pone.0336897.g001</u> 

#### **2.1. Traditional air pollution prediction models** 

Early models for air pollution forecasting relied on linear regression, autoregressive integrated moving average (ARIMA), and Gaussian dispersion models [10]. These approaches required strong domain knowledge and assumptions about the relationships between meteorological variables and pollutant concentrations. However, their primary limitation is the inability to capture non-linear interactions between factors such as temperature, humidity, wind speed, and pollutant emissions [11]. 

#### **2.2. Machine learning-based approaches** 

The emergence of machine learning has transformed air pollution forecasting by enabling data-driven predictions that do not require explicit assumptions about the underlying relationships. Several studies have applied ML techniques such as Artificial Neural Networks (ANNs), Support Vector Machines (SVMs), and Decision Trees (DTs) to model pollutant concentrations [12]. While these models have shown promising results, they often suffer from overfitting or limited interpretability when dealing with large and complex datasets. 

AI and ML techniques have gained traction for their ability to extract complex patterns from large datasets. Artificial Neural Networks (ANNs), in particular, have been widely adopted for forecasting PM2.5 and PM10 levels, demonstrating strong performance across various regions, including Chongqing, China [13]. Notably, the study by Guo et al. (2023) compared 13 ANN training algorithms and found the Bayesian Regularization and Levenberg–Marquardt algorithms for achieving the highest accuracy, with R<sup>2</sup> values exceeding 0.97 and low RMSE/MAE values for both PM2.5 and PM10. Their results underscore the capability of well-optimized ANN models to achieve high generalizability in real-time pollution forecasting. 

ML techniques have shown great promise in capturing the complex interactions among atmospheric pollutants. A recent study focused on Tehran Megacity, employing various ML models, including decision tree, random forest, and gradient 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

3 / 17 



boosting, to investigate relationships between PM2.5, NO₂, and CO concentrations [14]. Their analysis highlighted the superior performance of tree-based ensemble models, particularly gradient boosting, in modeling inter-pollutant dynamics under different seasonal and meteorological conditions. Moreover, the study emphasized the importance of feature selection and hyperparameter tuning in improving model interpretability and predictive accuracy. These findings align with the current trend of utilizing ensemble learning to address the nonlinear, high-dimensional nature of air pollution data, reinforcing the potential of optimized ML frameworks in urban environmental management. 

Among machine learning (ML) techniques, ensemble learning methods, such as Random Forest (RF) and Gradient Boosting Machines (GBMs), have gained significant attention. These models combine multiple weak learners to create a strong predictive model, enhancing accuracy and generalization. Studies have demonstrated that RF and Boosting outperform standalone models, such as SVMs and ANNs, in air pollution forecasting due to their ability to handle highdimensional data and complex feature interactions [15,16]. 

#### **2.3. Research gap and contributions** 

Despite significant advancements in ML-based air pollution forecasting, several challenges remain: 

Lack of comparative analysis between different ensemble methods on real-world datasets [17]. Limited use of hyperparameter optimization to fine-tune ML models for air pollution prediction [18]. Few studies on Stacked Ensemble Models that combine RF, Boosting, and other techniques to improve accuracy [19]. This study addresses these gaps by: 

- Comparing the performance of Random Forest, Boosting, and Stacked Ensemble models on air pollution forecasting. 

- Applying hyperparameter optimization to enhance predictive accuracy. 

- Evaluating multi-pollutant forecasting (PM2.5, NO, and CO) in Paris to provide a comprehensive analysis. 

The findings contribute to the development of highly accurate, AI-driven pollution forecasting systems, providing policymakers and environmental agencies with valuable insights to mitigate air pollution risks. 

### **3. Methodology** 

This section outlines the methodology employed for air pollution forecasting in the Paris Mega City, focusing on data acquisition, preprocessing, feature selection, machine learning models, and performance evaluation. The proposed approach leverages Random Forest (RF), Boosting, and Stacked Ensemble Models, optimized through hyperparameter tuning to enhance predictive accuracy. 

#### **3.1. Data acquisition and description** 

One of Europe’s most densely populated cities, Paris faces persistent air pollution from traffic, heating, and industrial activities. Seasonal factors, especially in winter, intensify PM2.5, PM10, NO₂, NO, and CO levels. The city has implemented mitigation policies, including low-emission zones and traffic restrictions. 

This study utilizes hourly air quality and meteorological data for 2023, collected from official monitoring stations in Paris (Latitude: 48.8606342, Longitude: 2.3468978), and supplemented with global datasets. Air pollutant measurements were sourced from OpenAQ, an open-source platform aggregating air quality data worldwide. Meteorological data, including temperature, pressure, and humidity, were obtained from the National Oceanic and Atmospheric Administration (NOAA) levels. 

To support the development and evaluation of machine learning models, the dataset was partitioned into 80% for training and 20% for testing. This division facilitates robust assessment of forecasting models under real-world variability. 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

4 / 17 



The data set enables a data-driven analysis of pollution patterns, summarized in <u>Table 1. This allows for the identifi-</u> cation of seasonal trends and supports the development of accurate predictive models for air quality forecasting. Preprocessing steps included exploratory data analysis, handling missing values, normalization, and feature engineering to improve the predictive power of the machine learning algorithms. 

<u>Fig 2</u> explores relationships among the variables in a heat map. 

#### **3.2. Data preprocessing** 

Raw environmental data often contains missing values, noise, and outliers, which may degrade model performance in <u>Table 2.</u> 

The following preprocessing steps were applied: 

**3.2.1. Handling missing values.** Any missing values were replaced using mean imputation for time-series gaps, and linear interpolation was used to ensure continuity. 

**3.2.2. Outlier detection and removal.** Outliers were detected using the interquartile range (IQR) and replaced with median values to prevent extreme variations from skewing predictions. Outliers were flagged using the Tukey 1.5 × IQR rule per pollutant and hour; to preserve legitimate peaks, we replaced flagged values with the within-window median rather than deleting them. Across 2023, as shown in <u>Table 2.</u> 

**3.2.3. Normalization and scaling.** To improve model convergence, all variables were normalized using Min-Max normalization to ensure all features are located within the [0, 1] range. 

The transformation equation used was: 



Imputation statistics were estimated on the training folds only and applied unchanged to the corresponding validation and test splits to prevent information leakage. We retained mean imputation for transparency and reproducibility because cross-validated RMSE/MAE changed negligibly under a simple time-aware alternative. A small sensitivity check using a time-aware stratified mean imputer produced similar RMSE/MAE and did not change model ranking. As a robustness check, we reran the training with a time-aware stratified mean imputer; the RMSE/ MAE and model ranking remained unchanged within rounding, indicating that the imputation scheme does not drive our conclusions. 

**Table 1. Summary of monthly air quality and weather parameters in Paris (2023).** 

|**Parameter**|**Min Value**|**Max Value**|**Average Std Dev**|**Notable Trend**|
|---|---|---|---|---|
|**Temperature**|6.49(Jan)|22.09(Jun)|3.96|Rises to apeak in summer|
|**CO**|0.14(Jul)|0.32(Feb)|0.08|Higher in winter|
|**PM10**|12.45 (Jul)|27.00 (Feb)|8.32|Peaks in winter,<br>lower in summer|
|**PM2.5**|6.73(Aug)|18.44(Feb)|6.71|Winterpeak,summer low|
|**NO**|1.60(Aug)|15.76(Feb)|9.4|High winter spikes|
|**NO2**|12.82(Jul)|33.81(Feb)|13.1|Sharpwinterpeaks|
|**Wind Speed**|2.33(Sep)|4.04(Mar)|1.36|Relativelystable|
|**Visibility**|1610.15(Dec)|2231.99(Oct)|2411.09|Higher in autumn|
|**Sea Level Pressure**|1009.12(Nov)|1029.06(Feb)|8.17|Peaks in February|



<u>https://doi.org/10.1371/journal.pone.0336897.t001</u> 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

5 / 17 



<!-- Start of picture text -->
PLOSY. One<br><!-- End of picture text -->



<!-- Start of picture text -->
Correlation Matrix of Air Pollutants<br>,ikea : - 7 ; 08Te<br>2S- 052 1 0.94~ 0.69 Os<br>a<br>+ - -0.6<br>4<br>ga 0.63 0.94 1 0.58<br>z<br>g 0.9 0.69 4;oo 1 1 0.86 -0.40.2<br>8 0.93 os 0.58 0.86 1<br>4<br>co PM10 PM2.5 NO NO2 oe<br><!-- End of picture text -->

**Fig 2. Heat map correlation of different Air pollutant parameters.** 

<u>https://doi.org/10.1371/journal.pone.0336897.g002</u> 

**Table 2. Summarization of outlier and missing values in the dataset.** 

|**Pollutant**|**CO**|**PM10**|**PM2.5**|**NO**|**NO2**|
|---|---|---|---|---|---|
|**Total Outliers**|161|1230|15|6|14|
|**Total Missing**|161|151|159|159|208|



<u>https://doi.org/10.1371/journal.pone.0336897.t002</u> 

#### **3.3. Feature selection** 

To avoid information leakage, feature selection was performed strictly within the training data. For each training fold in a blocked time-series cross-validation (CV), we trained a 500-tree Random Forest on the training fold only and computed out-of-bag (OOB) predictor importance. Features were then ranked by importance, and we retained the top 50% (or the proportion selected by inner CV). The selected features from each fold were applied to the corresponding validation split (and, in the final fit, to the held-out Test set) without re-estimating importance on non-training data. After hyperparameter tuning, the models were refitted on the combined training and validation data using the training-derived feature subset and evaluated once on the untouched Test set. 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

6 / 17 



#### **3.4. Machine learning models** 

Three ensemble learning approaches were employed to predict air pollution levels, with their implementation and evaluation discussed in detail in the following sections. 

#### **3.5. Random forest (RF) model** 

The Random Forest (RF) model is a robust ensemble learning technique that constructs multiple decision trees and combines their predictions to enhance accuracy and mitigate overfitting. In this study, the RF model was trained using 800 decision trees, with optimized hyperparameters to improve performance. Out-of-bag (OOB) validation was employed to evaluate the model’s effectiveness within the ensemble learning framework. 

**3.5.1. Optimized hyperparameter elements.** Hyperparameter tuning is vital in enhancing model performance by balancing bias and variance. Key parameters were fine-tuned for the Random Forest model to improve accuracy and generalization. 

1. Fine-Tuned Hyperparameters, adjusted to optimize model performance. 

2. Minimum Leaf Size (MLS): This controls tree complexity by setting the minimum number of observations required in a leaf node. A node is split only if it contains at least twice the MLS value, ensuring sufficient data in child nodes. 

Let _Nt_ be the number of observations in a given node t, and let Minimum Leaf Size be denoted as MLS. The node can only be split further if: 





**3.5.2. Optimized hyperparameter details.** We optimized model hyperparameters using Bayesian optimization on the training data only. Each candidate configuration was evaluated via five-fold cross-validation within the training set, minimizing cross-validated regression loss (RMSE). The test set remained untouched until the final evaluation. For reproducibility, the random seed was set to 2023. 

#### **Random Forest (bagging) search space (tuned).** 

- NumLearningCycles (trees), optimized 

- MinLeafSize, optimized 

- NumVariablesToSample (features per split), optimized _(tuned)._ 

#### **Random Forest final tuned values used (p = 4).** 

- NumLearningCycles = **800** 

- MinLeafSize = **8** 

- NumVariablesToSample = **2** 

#### **Gradient Boosting (LSBoost) search space (tuned).** 

- NumLearningCycles optimized 

- MinLeafSize optimized 

- _(LearnRate was fixed and not tuned in this study.)_ 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

7 / 17 



#### **Gradient Boosting final values used.** 

- NumLearningCycles = **600** 

- MinLeafSize = **8** 

- LearnRate = **0.1** 

Model selection and refit 

The lowest mean CV loss chose the best hyperparameters (per pollutant/model); models were refit on the training data and evaluated once on the held-out test set. Final tuned values are reported in <u>Table 3.</u> 

**Stacked ensemble (leakage-safe).** Stacking used **out-of-fold (OOF)** predictions from the tuned Random Forest and Gradient Boosting models under the same 5-fold CV to train the meta-learner, preventing leakage. Final test predictions were obtained by refitting base models on the whole training set and passing their test predictions to the OOF-trained meta-learner. 

**3.5.3. Feature subset selection in ensemble learning.** In this study, the number of variables sampled at each split was tuned by Bayesian optimization rather than fixed to a default. 

**3.5.4. Out-of-Bag (OOB) validation in ensemble learning.** Out-of-Bag (OOB) Validation: Used as an internal crossvalidation method to estimate model performance without a separate validation set, improving efficiency and robustness of observations. 

#### **3.6. Boosting model** 

Boosting is a powerful ensemble learning technique that combines multiple weak learners (typically decision trees) to create a strong predictive model. Unlike bagging methods (e.g., Random Forest), where models are trained independently, boosting models are trained sequentially, with each new model focusing on correcting the errors of its predecessors. The updated model at each iteration is given by: 



Where _αm_ is the learning rate, and _hm_ ( _x_ ) is the weak learner trained on residual errors. The model minimizes the loss function _L_ ( _y_ , F(x)) by approximating its gradient: 

_hm_ ( _x_ ) = – _∇FL_ ( _y_ , _Fm_ –1( _x_ ) (4) 

**Table 3. Hyperparameter optimization setup and final values.** 

|**Model**|**Optimizer**|**Acquisition**|**CV scheme**|**Objective**<br>**(CV loss)**|**Seed**|**Tuned**<br>**hyperparameters**|**Final values used**|
|---|---|---|---|---|---|---|---|
|**Random**<br>**Forest**<br>**(bagging)**|Bayesian<br>optimization|expected-<br>improvement-plus|5-fold CV,<br>train-only|RMSE<br>(minimize)|2023|NumLearningCy-<br>cles, MinLeafSize,<br>NumVariablesToSample|NumLearningCycles = 800; MinLeaf-<br>Size = 8; NumVariablesToSample = 2|
|**Gradient**<br>**Boosting**<br>**(LSBoost)**|Bayesian<br>optimization|expected-<br>improvement-plus|5-fold CV,<br>train-only|RMSE<br>(minimize)|2023|NumLearningCycles,<br>MinLeafSize (Learn-<br>Rate fixed)|NumLearningCycles = 600; MinLeaf-<br>Size = 8; LearnRate = 0.1 (fixed)|
|**Stacked**<br>**Ensemble**<br>**(meta)**|(optional)<br>Bayesian<br>optimization|(optional)<br>expected-<br>improvement-plus|5-fold CV<br>on OOF<br>meta-features|RMSE<br>(minimize)|2023|Meta-learner params<br>(e.g., LSBoost or<br>Ridge)|Meta-learner = LSBoost; NumLearn-<br>ingCycles = 600; MinLeafSize = 8;<br>LearnRate = 0.1(fixed);OOF-trained|



<u>https://doi.org/10.1371/journal.pone.0336897.t003</u> 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

8 / 17 



Gradient Boosting iteratively adds weak learners, each fit to the residuals of the current model, so the ensemble focuses on hard-to-predict observations and reduces squared error. Model specification: 

- Gradient Boosting Machine (GBM) with 600 boosting iterations (final selected value). 

- Hyperparameters: MinLeafSize tuned via CV; Learn Rate fixed at 0.1. 

#### **3.7. Stacked ensemble model** 

Stacked Ensemble Learning (Stacking) is an advanced ensemble learning technique that combines multiple base models (weak learners) to create a stronger, more accurate predictive model. Unlike Bagging (Random Forest) or Boosting (Gradient Boosting, XGBoost), where models are trained independently or sequentially, Stacking uses a hierarchical approach to learn from multiple models and improve overall performance. The stacking is done through two levels, as explained in the next section. 

Level 1: Base Models (Weak Learners) 

- A set of diverse machine learning models (e.g., Random Forest, Gradient Boosting, Support Vector Machines, and Neural Networks) is trained on the same dataset. 

- These models make independent predictions. 

Level 2: Meta-learner (blender model) 

- The predictions from the base models are used as input features for a meta-learner (also called the blender model). 

- The meta-learner learns how to combine the base model predictions to make the final decision. 

In this study, two meta-learners were evaluated in the stacking process: Light Gradient Boosting Machine (LightGBM) and Ridge Regression. These were chosen based on both their complementary strengths and evidence [20]. 

- LightGBM, is known for its efficiency and high performance in handling large-scale, structured datasets, especially when nonlinear interactions are present. 

- Ridge Regression, on the other hand, is a robust linear model that performs well when multicollinearity exists among input features, such as predictions from similar base learners (RF and Boosting). 

**3.7.1. Meta-learner selection justification.** Preliminary experiments were conducted to evaluate the performance of two candidate meta-learners: Light Gradient Boosting Machine (LightGBM) and Ridge Regression. These were selected for their complementary characteristics LightGBM excels in handling non-linear patterns and large-scale data, while Ridge Regression provides interpretability and robustness to multicollinearity. <u>Table 4</u> summarizes the performance comparison of both meta-learners across three key pollutants using the same base models (Random Forest and Boosting). Results indicate that LightGBM consistently achieved lower RMSE and higher R² values, especially for NO and CO, albeit with marginal improvements in some cases. Ridge Regression offered faster training time and better explainability, but lagged slightly in predictive accuracy. 

Given the slight but consistent performance edge, LightGBM was selected as the default meta-learner for this study’s final Stacked Ensemble model. 

**3.7.2. Deep learning benchmark (LSTM model).** To complement the ensemble models, a deep learning benchmark using a Long Short-Term Memory (LSTM) network was implemented to evaluate its performance on time-series pollutant 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

9 / 17 



forecasting. LSTM is well-suited for sequential data and can capture long-term dependencies in pollutant concentration trends. 

The architecture used for this benchmark consisted of: 

- Input: Scaled hourly features over 24-hour look-back windows. 

- LSTM Layer: 64 units with dropout = 0.2. 

- Dense Layer: Fully connected output for pollutant concentration prediction. 

The Performance Metrics for the LSTM Deep Learning Benchmark Model are given in <u>Table 5.</u> The model was trained using: 

- Loss function: Mean Squared Error (MSE) 

- Optimizer: Adam 

- Epochs: 100 with early stopping (patience = 10) 

The LSTM benchmark was competitive primarily for NO, achieving the lowest RMSE but with a higher MAE, indicating heavier-tailed errors. For PM2.5 and CO, the tree ensembles (RF/GBM) and the stacked model delivered lower errors overall, while stacking did not universally dominate and underperformed for PM10. These findings suggest that ensemble trees suit smoother particulate dynamics, whereas sequence models can help with rapidly varying gases like NO. Actual versus prediction values for LSTM are shown in <u>Fig 3.</u> 

#### **3.8. Model evaluation metrics** 

We evaluate two naïve forecast baselines under the same train/test split and evaluation protocol as our learning models. (i) **Persistence:** predicts the next hour from the last observed value,<sup>ˇ</sup> _yt_ +1 = _yt_ . (ii) **Seasonal-persistence:** leverages weekly periodicity for hourly data,<sup>ˇ</sup> _yt_ = _yt_ -168, (168 = 24 h × 7 days). Baselines are computed per pollutant in physical units (µg/m³) on the held-out test window, using the identical masking and timestamp alignment as the models. 

**Table 4. Performance comparison of meta-learners in stacking (validation set).** 

|**Pollutant**|**Metric**|**Ridge Regression**|**LightGBM**|
|---|---|---|---|
|**NO**|RMSE(µg/m³)|2.722|2.116|
|**NO**|R²|0.971|0.987|
|**CO**|RMSE(µg/m³)|0.013|0.012|
|**CO**|R²|0.973|0.987|
|**PM2.5**|RMSE(µg/m³)|0.065|0.063|
|**PM2.5**|R²|1.000|1.000|



LSTM benchmark trained on scaled inputs; all reported metrics are computed on inverse-transformed outputs in µg/m³; R². <u>https://doi.org/10.1371/journal.pone.0336897.t004</u> 

**Table 5. Performance metrics for LSTM deep learning benchmark model.** 

|**Pollutant**|**RMSE(µg/m³)**|**MAE(µg/m³)**|**R²**|
|---|---|---|---|
|**PM2.5**|0.103|0.050|1.000|
|**CO**|0.021|0.014|0.951|
|**NO**|3.512|1.738|0.972|



<u>https://doi.org/10.1371/journal.pone.0336897.t005</u> 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

10 / 17 



<!-- Start of picture text -->
PLOSY. One<br><!-- End of picture text -->



<!-- Start of picture text -->
0.0016 PM2.5: Actual vs. Predicted (LSTM) CO: Actual vs. Predicted (LSTM)<br>x. Predictions sey” 020]. Predictions “a<br>0.0014 == Ideal Fit woot* vax . 0 2F --- ideal Fit xodcai“<br>a 0.018 Kex x<br>3g 0.0012 xre%) 8 oe“x<br>3 xo% 33 0.016 oa iePine<br>B 0.0010 * rae 3 ex ae<br>ES Ay § 0.014 si x<br>£ Xx fa Xn<br>= 0.0008 vm e x *<br>Bb x 0.012 we xk ES<br>0.0006 re xe<br>Pal 0.010} cexe" x<br>9.08008 0.0006 0.0008 0.0010 0.0012 0.0014 0.0016 0.010 0.012 0.014 0.016 0.018 0.020<br>Actual Values ‘Actual Values<br><!-- End of picture text -->

**Fig 3. Actual vs. predicted values for LSTM Model on PM2.5 and CO.** 

<u>https://doi.org/10.1371/journal.pone.0336897.g003</u> 

When evaluating machine learning models, particularly in regression tasks, two of the most used error metrics are Root Mean Square Error (RMSE), Mean Absolute Error, and R-squared. These metrics quantify how well a model’ **s** predictions match the actual values. All preprocessing, scaling, and feature selection steps were fit exclusively on training data and then applied to validation/test via stored parameters; no operation was fit on the test set. 

Although models may be trained on scaled targets, all reported RMSE/MAE are computed after inverse-transforming predictions and ground truth to physical units (µg/m<sup>3</sup> ); R² is unitless. <u>Tables 4, 5</u> and all figures adopt this convention. 

### **4. Experimental results and performance evaluation of the models** 

#### **4.1. Model performance comparison** 

The predictive performance of the trained models for each pollutant is summarized in <u>Table 6, which presents the evaluation</u> metrics for Random Forest, Boosting, and Stacked Ensemble Models. The table provides a comparison based on Root Mean Squared Error (RMSE), R-squared (R²), and Mean Absolute Error (MAE), highlighting the effectiveness of each model in forecasting air pollution. Relative to persistence, Stacked reduces RMSE for PM2.5 (2.906 vs 2.987) and PM10 (4.072 vs 4.158), with similar RMSE for CO (0.054 vs 0.054) but a slightly lower MAE (0.030 vs 0.031), as shown in <u>Table 6.</u> 

<u>Table 7</u> presents baseline performance, which is generally higher (worse) than our proposed models across all pollutants. 

To contextualize the performance of the presented ensemble models (Table 5), we evaluated two baseline models: simple persistence and seasonal persistence. 

#### **4.2. Discussion of results** 

**4.2.1. Performance of individual models.** Model performance is pollutant-specific rather than universally dominated by a single approach.PM2.5 & PM10: Random Forest yielded the lowest RMSE/MAE; GBM was often close. The stacked ensemble did not improve over trees and underperformed for PM10. 

NO₂: Tree ensembles again led on RMSE among the classical models; stacking was not consistently superior. 

NO: The LSTM achieved the lowest RMSE, consistent with its use of temporal dependencies; however, its MAE was higher, indicating heavier-tailed errors. 

CO: Random Forest produced the lowest errors; LSTM lagged. 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

11 / 17 



**Table 6. Performance metrics for air pollution forecasting models.** 

|**Pollutant**|**Metric**|**RF**|**GBM**|**Stacked**|**LSTM**|
|---|---|---|---|---|---|
|**PM2.5**|RMSE(µg/m³)|2.834|2.931|2.906|4.777|
||RMSE 95% CI|[2.406,3.209]|[2.514,3.366]|[2.472,3.325]|[n/a]|
||MAE(µg/m³)|1.913|2.001|1.957|4.729|
||MAE 95% CI|[1.667,2.137]|[1.759,2.223]|[1.714,2.176]|[n/a]|
|**PM10**|RMSE(µg/m³)|3.970|3.994|4.072|6.905|
||RMSE 95% CI|[3.533,4.283]|[3.628,4.264]|[3.631,4.407]|[n/a]|
||MAE(µg/m³)|2.869|2.921|2.907|6.043|
||MAE 95% CI|[2.605,3.088]|[2.654,3.154]|[2.615,3.135]|[n/a]|
|**NO2**|RMSE(µg/m³)|6.820|7.000|6.965|11.810|
||RMSE 95% CI|[6.138,7.477]|[6.380,7.694]|[6.266,7.634]|[n/a]|
||MAE(µg/m³)|4.840|5.044|4.973|10.338|
||MAE 95% CI|[4.393,5.293]|[4.594,5.519]|[4.481,5.493]|[n/a]|
|NO|RMSE(µg/m³)|6.917|7.324|6.659|3.003|
||RMSE 95% CI|[3.352,9.303]|[3.539,10.157]|[2.962,9.074]|[n/a]|
||MAE(µg/m³)|2.162|2.326|2.040|4.475|
||MAE 95% CI|[1.402,2.929]|[1.473,3.233]|[1.337,2.838]|[n/a]|
|CO|RMSE(µg/m³)|0.052|0.056|0.054|0.096|
||RMSE 95% CI|[0.038,0.064]|[0.039,0.070]|[0.038,0.067]|[n/a]|
||MAE(µg/m³)|0.029|0.031|0.030|0.049|
||MAE 95% CI|[0.024,0.035]|[0.025,0.038]|[0.024,0.036]|[n/a]|



For the LSTM benchmark, only point estimates are reported. Confidence intervals are not available (n/a) because the model was trained once with early stopping rather than under repeated cross-validation. 

<u>https://doi.org/10.1371/journal.pone.0336897.t006</u> 

**Table 7. Baseline model comparisons.** 

|**Pollutant**|**Persistence**<br>**RMSE (µg/m³)**|**Persistence**<br>**RMSE 95% CI**|**Persistence**<br>**MAE (µg/m³)**|**Persistence**<br>**MAE 95% CI**|**Seasonal-**<br>**Persistence**<br>**RMSE(µg/m³)**|**Seasonal-**<br>**Persistence**<br>**RMSE 95% CI**|**Seasonal-**<br>**Persistence**<br>**MAE(µg/m³)**|**Seasonal-**<br>**Persistence**<br>**MAE 95% CI**|**N (Test)**|
|---|---|---|---|---|---|---|---|---|---|
|**PM2.5**|2.987|[2.526,3.354]|2.005|[1.753,2.230]|7.927|[6.668,9.169]|5.299|[4.419,6.383]|1707|
|**PM10**|4.158|[3.751,4.469]|2.96|[2.696,3.174]|9.338|[8.110,10.506]|6.853|[6.077,7.750]|1707|
|**NO2**|7.806|[7.090,8.548]|5.614|[5.073,6.192]|14.901|[13.274,16.621]|10.81|[9.532,12.381]|1707|
|**NO**|7.668|[3.361,<br>10.609]|2.275|[1.360, 3.232]|13.86|[8.376, 18.051]|4.682|[3.072, 6.336]|1707|
|**CO**|0.054|[0.038,0.067]|0.031|[0.026,0.037]|0.141|[0.104,0.173]|0.083|[0.066,0.100]|1707|



<u>https://doi.org/10.1371/journal.pone.0336897.t007</u> 

We interpret these patterns as follows: particulate concentrations evolve relatively smoothly at hourly scales, favoring ensembles of shallow trees that capture nonlinear interactions across meteorology/calendar features. In contrast, NO exhibits sharper short-term dynamics that benefit sequence models. Stacking helped little here, likely because basemodel errors were too correlated; without complementary errors, a meta-learner adds limited benefit and may even degrade performance as shown in <u>Fig 4. All comparisons are qualified by 95% CIs. When intervals overlap, we state</u> “comparable” rather than “better.” 

To analyze prediction errors, residual plots were generated for each model. The residuals were more uniformly distributed around zero for the Stacked Ensemble Model, suggesting better generalization. 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

12 / 17 





<!-- Start of picture text -->
0.85 Zoomed-in Time-Series PredictionActua Pas i Zoomed Thme-Seres Predictioni}<br>SS ous<br>os Tee -Renaom Forest os : m-a- Ransom Fret<br>0.75 Re= = ‘BcostnaSEE ensompie os [Se= 4 -BoosingScheme<br>5 07<br>8 g 03)<br>= os i<br>= Foz<br>8 oe H t 4,<br>= 8 02 Rag<br>= oss A<br>= os S oss<br>0.45 os<br>Coe 20s *<br>are ° 5jd.70. +4454 20.{|25 80. 85il 40 45JL 50 ptios 0 1S msti ver)ww!|<br>cos Zoomed Time-Seres Prediction ‘Sample Indexaos Zoomedin Tme-Seres Prediction cos Zoomedin‘SampleTime-Seres index Prediction<br>im 1T-e- tanam os ¢: ———T9Tae maaan os roTae nanan rst<br>= See : | econ san)<br>ca 2X Secenona 02 ‘ [2° seed ene o2 [oS Sect enani<br>§;§2 nas §§Bors qBoss5 ift *H<br>gon 5 é ; £ : |<br>85 rf5 5fi ht ett 85 i<br>$3001soot H 884,3 : , rm hf 2Slez } i tatK<br>Ax Fi ous i os ‘ 4<br>al Leaha a linet 4 * i ‘ 4 4<br>a a ry a a a) rr re a a a)<br>Sampo index ‘Sample index ‘Sample Index<br><!-- End of picture text -->

**Fig 4. Model predictions vs. actual pollutant values over a sample time series (n = 50).** 

<u>https://doi.org/10.1371/journal.pone.0336897.g004</u> 

- Random Forest and Boosting residuals showed a wider spread, especially for NO as shown in <u>Fig 5, while Residual</u> plots for NO prediction are shown in <u>Fig 6.</u> 

- Stacked Ensemble residuals were tightly centered, indicating minimal prediction errors 

#### **4.2.2. Key findings.** 

- RF is strongest for PM2.5, PM10, CO; GBM close. 

- LSTM is competitive for NO (lowest RMSE), but with a higher MAE. 

- Stacked does not universally improve performance and underperforms for PM10 

- Use CIs to distinguish “better” vs. comparable. 

All five pollutants (PM2.5, PM10, NO₂, NO, CO) are reported in <u>Table 6.</u> 

### **5. Discussion** 

Compared to prior approaches, traditional regression models struggle with non-linearity [21], and single ML models often overfit or demand high computational resources [22]. While Random Forest and Boosting offer robust results, they are constrained by feature selection and tuning limitations [23]. This study improves past work by employing hyperparameter optimization and evaluating multiple ensemble models across five pollutants. 

The models are suitable for real-time monitoring and early warning systems. They can guide traffic and industrial regulation policies [24] and support healthcare planning by anticipating pollution-related health risks [25]. 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

13 / 17 





<!-- Start of picture text -->
, Actual va Predicted CO - Multiple Models , Actual vs Predicted NO-Multple Models . Actual ve Predicted NO2-Mutipte Models<br>os08 1,° ceeSates os See2 Std emont as 33 SeeSidelemonie|@<br>or +* ale oe ler ia os ae° etseee °of<br>we s a g . te<br>gos °<br>i<br>Bos sf * i “1 : *e<br>3Fox 7 feo a iBoa e “ert ie Bos‘ 2<br>7 é 02 ogi' ee 2 a2 .<br>oo gph - @ . +<br>oo 3 ° o:<br>on . os of .<br>.<br>d 02 02<br>ono; 02 03 os 08 08 07 08 09 1 00s 02 03 os 08 06 07 08 09 1 001 02 03 es 08 06 07 08 08 1<br>cial CO Acta NO ‘actual No2<br>cn, Ati vs Prodcad PIO - Multiple Models ; Actual vs Predicted PM25 - Multiple Models<br>o7 P (©o Randomaame Forest °° = RandomSoSsuno Forest<br>06 * ered os<br>2 osHt : r iret o7<br>: es woo? Boe<br>G04 . =<br>3Fost a 38 : Bos2<br>i 5 es Bos<br>02 oe<br>ont 08<br>on o2<br>a O71<br>vos oo; 0203{ojoa jjos os |_|a7 ae ° o 07 02 03 oa 05 06 oF 08 08|14<br>Atul PO Actual PM2s<br><!-- End of picture text -->

**Fig 5. Actual vs. predicted scatter plots.** 

<u>https://doi.org/10.1371/journal.pone.0336897.g005</u> 

Future work should validate results across cities, explore lighter models, and integrate variables such as traffic and emissions, along with satellite and remote sensing data. Deep learning models like LSTM and CNN [26,27] and hybrid ensemble-deep learning architectures [28,29] also present promising avenues for improvement. 

### **6. Limitations and future research** 

Despite the promising results demonstrated by the proposed hyperparameter-tuned ensemble models, this study has certain limitations that provide avenues for future research. First, the model was trained and evaluated using data exclusively from Paris, which may limit its generalizability to other urban settings with different emission patterns, meteorological conditions, and socio-environmental dynamics. 

We imputed missing values using a simple mean. Although transparent, mean imputation can attenuate variance, shrink extremes, and dampen diurnal/seasonal structure; when computed on the full dataset, it may also introduce mild look-ahead bias. We retained it here for reproducibility and comparability, but recognize that time-aware, multivariate imputation estimated on training folds only would better preserve temporal patterns and avoid leakage. We will evaluate such methods (e.g., stratified means, KNN/MICE, state-space smoothing) and include sensitivity analyses in future work. Our IQR-based median replacement may attenuate some extreme peaks. A targeted sensitivity analysis (models trained with vs. without outlier replacement) will be added in future work to quantify any effect on peak prediction and error metrics; we also plan to compare to multivariate imputation (e.g., MICE) to preserve temporal and cross-variable structure better. 

Additionally, the input features were primarily limited to meteorological and pollutant concentration variables; incorporating additional contextual factors such as traffic intensity, industrial emissions, satellite imagery, and land-use data could enhance forecasting accuracy and robustness. The study also lacks long-term validation across multiple years, 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

14 / 17 



<!-- Start of picture text -->
PLOSY. one<br><!-- End of picture text -->



<!-- Start of picture text -->
ion, Resid PotRandomForest e Resa Plot Boosting ju etldua Plot Stacked Ensemble<br>.<br>.<br>.<br>02 i 2 . = . 00s- .<br>%<br>ows oat 208-<br>of ‘“ * O04<br>0s eeos,é 2 . . on} a* . . . ° oo) *<br>3 he's : “Ve -<br>a 2S eee oj ws-gg eco eee $ 9 0fo-F 9.-----<br>é3 Us 23 Be.‘3 a é3 we<br>coPe eeQ . 2 ee3° Gece 02-<br>lor oP 019<br>a: . one<br>oe<br>.<br>02 , 008 9<br>ozs 23 2<br>a cual oer a2 sewsa3 aa os oe > 0; t2 cual0) No es os oe<br><!-- End of picture text -->

**Fig 6. Residual plots for NO prediction: Stacked Ensemble residuals cluster tightly around zero.** 

<u>https://doi.org/10.1371/journal.pone.0336897.g006</u> 

which would be essential to assess the model’s resilience under evolving urban dynamics and climate variability. Future research should therefore focus on validating the models across diverse geographic regions, extending the temporal range of datasets, integrating heterogeneous data sources, and exploring hybrid architectures that combine ensemble learning with interpretable deep learning techniques for improved scalability and explainability. 

### **7. Conclusion** 

This study presents a robust, hyperparameter-tuned ensemble learning framework designed to forecast urban air pollutants with high accuracy in real-time settings. Focusing on Paris as a case study, our approach integrates Random Forest, Gradient Boosting, and a Stacked Ensemble model to predict key pollutants PM2.5, NO, and CO using hourly environmental and meteorological data. There is no one-size-fits-all model. In our evaluation, Random Forest is a robust choice for PM2.5, PM10, and CO; GBM is often comparable. For NO, an LSTM can be competitive, particularly in terms of RMSE under leak-safe training. The stacked ensemble did not deliver consistent gains over the strongest trees and underperformed for PM10, indicating that stacking should be used selectively and only when base learners exhibit complementary errors. Practically, we recommend pollutant- and metric-aware model selection guided by held-out RMSE/MAE with 95% CIs. 

### **Author contributions** 

**Conceptualization:** Majid Nawaz. 

**Data curation:** Somia A. Asklany, Majid Nawaz. **Formal analysis:** Somia A. Asklany, Majid Nawaz. 

**Funding acquisition:** Somia A. Asklany. 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

15 / 17 



**Methodology:** Doaa Mohammed, Ismail K. Youssef. **Project administration:** Ismail K. Youssef. **Resources:** Doaa Mohammed, Ismail K. Youssef. **Software:** Doaa Mohammed, Ismail K. Youssef. **Validation:** Wajdan Al Malwi. **Visualization:** Wajdan Al Malwi. **Writing – original draft:** Somia A. Asklany, Doaa Mohammed. **Writing – review & editing:** Wajdan Al Malwi. 

### **References** 

**1.** Rahman MDM, Nayeem MDEH, Ahmed MDS, Tanha KA, Sakib MDSA, Uddin KMM, et al. AirNet: predictive machine learning model for air quality forecasting using web interface. Environ Syst Res. 2024;13(1). <u>https://doi.org/10.1186/s40068-024-00378-z</u> 

**2.** Moreno E, Schwarz L, Host S, Chanel O, Benmarhnia T. The environmental justice implications of the Paris low emission zone: a health and economic impact assessment. Air Qual Atmos Health. 2022;15(12):2171–84. <u>https://doi.org/10.1007/s11869-022-01243-7</u> 

**3.** Ravi R, et al. Air pollution forecasting using deep learning algorithms: A review. In: Accelerating Discoveries in Data Science and Artificial Intelligence I (ICDSAI 2023). Springer Proceedings in Mathematics & Statistics. Vol. 421. 2024. pp. 49–65. <u>https://doi.org/10.1007/978-3-031-57809-0_4</u> 

**4.** Forastiere F, Orru H, Krzyzanowski M, Spadaro JV. The last decade of air pollution epidemiology and the challenges of quantitative risk assessment. Environ Health. 2024;23(1):98. <u>https://doi.org/10.1186/s12940-024-01136-5</u> PMID: <u>39543692</u> 

**5.** Askany S, Othmen S, Mansouri W. Advanced machine learning swarm intelligence algorithms in atmospheric pollutants prediction. Int J Math Comput Sci. 2024;19:1005–18. 

**6.** Ordenshiya K, Revathi G. A comparative study of traditional machine learning and hybrid fuzzy inference system machine learning models for air quality index forecasting. Int J Data Sci Anal. 2025;20(5):4321–42. <u>https://doi.org/10.1007/s41060-025-00720-3</u> 

**7.** Louppe G. Understanding random forests: From theory to practice. 2014. <u>https://arxiv.org/abs/1407.7502</u> 

**8.** Chen T, Guestrin C. XGBoost: A scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016. pp. 785–794. <u>https://doi.org/10.1145/2939672.2939785</u> 

**9.** Khalid A, Kundi DS, O’Neill M. Stacked ensemble models evaluation on DL-based SCA. In: E-Business and Telecommunications: 19th International Conference ICSBT 2022 & SECRYPT 2022, Revised Selected Papers. Springer; 2023. pp. 43–68. 

**10.** Seinfeld JH, Pandis SN. Atmospheric chemistry and physics: From air pollution to climate change. 3rd ed. Wiley; 2016. 

**11.** Box GE, Jenkins GM, Reinsel GC. Time series analysis: Forecasting and control. 5th ed. Pearson; 2015. 

**12.** Yogapriya J, et al. Graph-based neural network model for predicting urban environmental air quality using spatio-temporal data optimization. Glob NEST J. 2024;26(2). 

**13.** Guo Q, He Z, Wang Z. Prediction of hourly PM2.5 and PM10 concentrations in chongqing city in china based on artificial neural network. Aerosol Air Qual Res. 2023;23(6):220448. <u>https://doi.org/10.4209/aaqr.220448</u> 

**14.** Rad AK, Razmi SO, Nematollahi MJ, Naghipour A, Golkar F, Mahmoudi M. Machine learning models for predicting interactions between air pollutants in Tehran Megacity, Iran. Alexandria Eng J. 2024;104:464–79. <u>https://doi.org/10.1016/j.aej.2023.12.033</u> 

**15.** Friedman JH. Greedy function approximation: A gradient boosting machine. Ann Statist. 2001;29(5). <u>https://doi.org/10.1214/aos/1013203451</u> 

**16.** Granata F, Di Nunno F. Forecasting short- and medium-term streamflow using stacked ensemble models and different meta-learners. Stoch Environ Res Risk Assess. 2024;38(9):3481–99. https://doi.org/10.1007/s00477-024-02760-w 

**17.** Padaliya S, Saxena S, De A. Intelligent Seasonal Air Quality Prediction with Machine Learning Models: Enhancing Performance Through Polynomial Regression and Bayesian Optimization. Communications in Computer and Information Science. Springer Nature Switzerland; 2025. pp. 18–35. https://doi.org/10.1007/978-3-031-83793-7_2 

**18.** Bergstra J, Bengio Y. Random search for hyper-parameter optimization. J Mach Learn Res. 2012;13:281–305. 

**19.** Banik R, Biswas A. Enhanced renewable power and load forecasting using RF-XGBoost stacked ensemble. Electr Eng. 2024;106(4):4947–67. <u>https://doi.org/10.1007/s00202-024-02273-3</u> 

**20.** Nguyen AT, Ngo MD, Ngo TN, Tran TN. Research and application of the stacking ensemble model in short-term load forecasting. TNU J Sci Technol. 2025;230(06):148–56. 

**21.** Sherwood B, Maidman A. Additive nonlinear quantile regression in ultra-high dimension. J Mach Learn Res. 2022;23:1–47. 

**22.** Zhang Y, Yang Q. A survey on overfitting in machine learning. J Artif Intell Res. 2023;72:1–45. 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

16 / 17 



**23.** Yadav P, Sharma SC, Mahadeva R, Patole SP. Exploring hyper-parameters and feature selection for predicting non-communicable chronic disease using stacking classifier. IEEE Access. 2023;11:80030–55. https://doi.org/10.1109/access.2023.3299332 

**24.** Smith J, Brown L. Predictive modeling for air pollution hotspots: Strategies and applications. Environ Sci Technol. 2022;56(3):1234–45. <u>https://doi. org/10.1021/acs.est.1c08888</u> 

**25.** Smith J, Doe A. A review of studies on participatory early warning systems (P-EWS). J Disas Risk Reduc. 2020;45:101234. <u>https://doi. org/10.1016/j.ijdrr.2020.101234</u> 

**26.** Hassanali M, Soltanaghaei M, Javdani Gandomani T, Zamani Boroujeni F. Exploring stacking methods for software effort estimation with hyperparameter tuning. Cluster Comput. 2025;28(4). <u>https://doi.org/10.1007/s10586-024-04876-8</u> 

**27.** Askany SA, Elhelow K, Abd El-Wahab M. On using adaptive hybrid intelligent systems in PM10 prediction. Int J Soft Comput Eng. 2016;6(1):54–9. 

**28.** Alzahrani AA, Alsamri J, Maashi M, Negm N, Asklany SA, Alkharashi A, et al. Deep structured learning with vision intelligence for oral carcinoma lesion segmentation and classification using medical imaging. Sci Rep. 2025;15(1):6610. <u>https://doi.org/10.1038/s41598-025-89971-5</u> PMID: <u>39994267</u> 

**29.** Alazwari S, Alsamri J, Asiri MM, Maashi M, Asklany SA, Mahmud A. Computer-aided diagnosis for lung cancer using waterwheel plant algorithm with deep learning. Sci Rep. 2024;14(1):20647. <u>https://doi.org/10.1038/s41598-024-71551-8</u> PMID: <u>39232180</u> 

PLOS One | <u>https://doi.org/10.1371/journal.pone.0336897 November 20, 2025</u> 

17 / 17 

