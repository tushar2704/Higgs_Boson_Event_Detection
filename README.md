# Road Traffic Severity Classification by Tushar Aggarwal 
### Introduction About the Data :

**This data set is collected from Addis Ababa Sub-city police departments for master's research work. The data set has been prepared from manual records of road traffic accidents of the year 2017-20. All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms.**

**Problem Statement: The target feature is Accident_severity which is a multi-class variable. The task is to classify this variable based on the other 31 features step-by-step by going through each day's task. Your metric for evaluation will be f1-score.**

Dataset Source Link :
[NARCIS](https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591)

# Demo:
```
streamlit run RTS_app.py
```
 Pip install libraries
```
pip install -r requirements.txt
```
# Deployed Link:
### Streamllit:
 ```
https://tushar2704-road-traffic-severity-rts-app-sf35uz.streamlit.app/
 ```
### Render:
 ```
https://rts-app.onrender.com
 ```



### It is observed that the categorical variables 'cut', 'color' and 'clarity' are ordinal in nature

### Check this link for details : []()

# AWS Deployment Link :

AWS Elastic Beanstalk link : []()

# Screenshot of UI

![HomepageUI](./Screenshots/HomepageUI.jpg)

# YouTube Video Link

Link for YouTube Video : Click the below thumbnail to open 



# AWS API Link

API Link : []()

# Postman Testing of API :

![API Prediction](./Screenshots/APIPrediction.jpg)

# Approach for the project 

1. Data Ingestion : 
    * In Data Ingestion phase the data is first read as csv. 
    * Then the data is split into training and testing and saved as csv file.

2. Data Transformation : 
    * In this phase a ColumnTransformer Pipeline is created.
    * for Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
    * for Categorical Variables SimpleImputer is applied with most frequent strategy, then ordinal encoding performed , after this data is scaled with Standard Scaler.
    * This preprocessor is saved as pickle file.

3. Model Training : 
    * In this phase base model is tested . The best model found was catboost regressor.
    * After this hyperparameter tuning is performed on catboost and knn model.
    * A final VotingRegressor is created which will combine prediction of catboost, xgboost and knn models.
    * This model is saved as pickle file.

4. Prediction Pipeline : 
    * This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.

5. Flask App creation : 
    * Flask app is created with User Interface to predict the gemstone prices inside a Web Application.

# Exploratory Data Analysis Notebook

Link : [EDA Notebook](./notebook/1_EDA_Gemstone_price.ipynb)

# Model Training Approach Notebook

Link : [Model Training Notebook](./notebook/2_Model_Training_Gemstone.ipynb)

# Model Interpretation with LIME 

Link : [LIME Interpretation](./notebook/3_Explainability_with_LIME.ipynb)






"""""""""""""""""""""""""""""""
## Loan Approval Prediction - Random Forest Model and Deployed Web App

## Project Purpose: 
To demonstrate a full ML project from scratch to deployment to a web app.

## Business Case: 
To create a model that accurately predicts loan approval and automates the manual approval process. 

## Goal: 
To accurately predict loan approvals without falsely approving too many loans that should be denied (minimize False Positives). 

## Deliverable: 
A web app that end users can utilize to predict loan approvals using a Random Forest model on the backend. 

    1. Web App: https://jessramos2-loan-approval-random-forest-web-streamlit-app-47jl27.streamlitapp.com/
    
  <img src="WebAppPic.jpg" width="430" height="400">

### In Repository:

1. Data & data dictionary

<img src="DataDictionary.jpg" width="350" height="250">

2. Random Forest Model Creation (Loan Approval Model.ipynb)

3. Sample decision tree from random forest
    
 ![](DecisionTree.jpg)

4. Web App Python Code (streamlit_app.py)

5. Web app link and screenshot


## Results and Model Evaluation: 

From a business perspective, we want to avoid predicting a positve loan approval when it was actually denied (False Positives), so ***Precision*** will be our best measures for the model. This will ensure that we are accurately predicting loan approvals and that predicted approvals are actual approvals (precision). 

Since this model produces very similar precision scores on both the training and testing data, it appears to be the best fit to maximize predictive power on the training dataset without overfitting and sacrificing predictability on the testing data. 

**Precision:** 
Precision on the testing data is ~78%, which means that we don't have a large amount of False Positives. This is great, because as a business, we want to avoid predicting loans approvals that will have to be denied later. 


**Accuracy:**
Accuracy on the testing data is ~80% which means that the model correctly predicts 4/5 of the loans. 


**Recall:**
 Recall on the testing data is 100% which means that the model accurately predicts all True Positives. This means that we will not miss out on any potential loan approvals (and revenue). 


**F1 Score:**
The F1 score on the testing data is ~88%, which is great since it takes into account both False Positives and False Negatives. 


## Business Impact: 

End users will be able to use the web app built off of this model to predict loan approvals right in front of the borrower. There will be no missed revenue opportunities since the model captures all true approvals (recall is 100%), and only a small portion of borrowers predicted to be approved will actually be denied. This will speed up the manual approval process and allow the company to process more loans in less time, resulting in more clients and revenue. 


### Next Steps: Monitor performance and retrain model with more data as more data becomes available. 
# Game-Of-Thrones-Text-Gen
# Higgs_Boson_Event_Detection
# Higgs Boson Event Detection
 
## Backstory

**Particle accelerators.** To probe into the basic questions on how matter, space and time work and how they are structured, physicists focus on the simplest interactions (for example, collision of [**subatomic particles**](https://en.wikipedia.org/wiki/Subatomic_particle)) at very high energy. [**Particle accelerators**](https://en.wikipedia.org/wiki/Particle_accelerator) enable physicists to explore the fundamental nature of matter by observing subatomic particles produced by high-energy collisions of [**particle beams**](https://en.wikipedia.org/wiki/Particle_beam). The experimental measurements from these collisions inevitably lack precision, which is where [**machine learning**](https://en.wikipedia.org/wiki/Machine_learning) (ML) comes into picture. The research community typically relies on standardized machine learning software packages for the analysis of the data obtained from such experiments and spends a huge amount of effort towards improving statistical power by extracting features of significance, derived from the raw measurements.

**Higgs boson.** The [**Higgs boson**](https://en.wikipedia.org/wiki/Higgs_boson) [**particle**](https://en.wikipedia.org/wiki/Elementary_particle), also called the **God particle** in mainstream media, is the final ingredient of the [**standard model**](https://en.wikipedia.org/wiki/Standard_Model) of [**particle physics**](https://en.wikipedia.org/wiki/Particle_physics), which sets the rules for the subatomic particles and forces. The [**elementary particles**](https://en.wikipedia.org/wiki/Elementary_particle) are supposed to be massless at very high energies, but some of them can acquire mass at low-energies. The mechanism of this acquiring remained an enigma in theoretical physics for a long time. In $1964$, [**Peter Higgs**](https://en.wikipedia.org/wiki/Peter_Higgs) and others proposed a [**mechanism**](https://en.wikipedia.org/wiki/Higgs_mechanism) that theoretically explains the [**origin of mass of elementary particles**](https://en.wikipedia.org/wiki/Mass_generation). The mechanism involves a **field**, commonly known as [**Higgs field**](https://en.wikipedia.org/wiki/Higgs_mechanism#Structure_of_the_Higgs_field), that the paricles can interact with to gain mass. The more a particle interacts with it, the heavier it is. Some particles, like [**photon**](https://en.wikipedia.org/wiki/Photon), do not interact with this field at all and remain massless. The Higgs boson particle is the associated particle of the Higgs field (all fundamental fields have one). It is essentially the physical manifestation of the Higgs field, which gives mass to other particles. The detection of this elusive particle waited almost half a century since its theorization!

**The discovery.** On 4th July 2012, the [**ATLAS**](https://home.cern/science/experiments/atlas) and [**CMS**](https://home.cern/science/experiments/cms) experiments at [**CERN**](https://en.wikipedia.org/wiki/CERN)'s [**Large Hadron Collider**](https://en.wikipedia.org/wiki/Large_Hadron_Collider) announced that both of them had observed a new particle in the mass region around 125 GeV. This particle is consistent with the theorized Higgs boson. This experimental confirmation earned [**François Englert**](https://en.wikipedia.org/wiki/Fran%C3%A7ois_Englert) and Peter Higgs [**The Nobel Prize in Physics 2013**](https://www.nobelprize.org/prizes/physics/2013/summary/)
> "for the theoretical discovery of a mechanism that contributes to our understanding of the origin of mass of subatomic particles, and which recently was confirmed through the discovery of the predicted fundamental particle, by the ATLAS and CMS experiments at CERN's Large Hadron Collider."

**Giving mass to fermions.** There are many different processes through which the Higgs boson can decay and produce other particles. In physics, the possible transformations a particle can undergo as it decays are referred to as [**channels**](https://atlas.cern/glossary/decay-channel). The Higgs boson has been observed first to decay in three distinct decay channels, all of which are [**boson**](https://en.wikipedia.org/wiki/Boson) pairs. To establish that the Higgs field provides the interaction which gives mass to the fundamental [**fermions**](https://en.wikipedia.org/wiki/Fermion) (particles which follow the [**Fermi-Dirac statistics**](https://en.wikipedia.org/wiki/Fermi%E2%80%93Dirac_statistics), contrary to the bosons which follow the [**Bose-Einstein statistics**](https://en.wikipedia.org/wiki/Bose%E2%80%93Einstein_statistics)) as well, it has to be demonstrated that the Higgs boson can decay into fermion pairs through direct [**decay**](https://en.wikipedia.org/wiki/Particle_decay) modes. Subsequently, to seek evidence on the decay of Higgs boson into fermion pairs (such as [**tau leptons**](https://simple.wikipedia.org/wiki/Tau_lepton) $(\tau)$ or [**b-quarks**](https://en.wikipedia.org/wiki/Bottom_quark)) and to precisely measure their characteristics became one of the important lines of enquiry. Among the available modes, the most promising is the decay to a pair of tau leptons, which balances a modest branching ratio with manageable backgrounds.

**The first evidence of $h \to \tau^+\tau^-$ decays [was recently reported](https://cds.cern.ch/record/1632191), based on the full set of proton–proton collision data recorded by the ATLAS experiment at the LHC during $2011$-$2012$. Despite the consistency of the data with $h \to \tau^+\tau^-$ decays, it could not be ensured that the statistical power exceeds the $5\sigma$ threshold, which is the required standard for claims of discovery in high-energy physics community.**

<figure>
    <img src = "https://raw.githubusercontent.com/sugatagh/Higgs-Boson-Event-Detection/main/Image/atlas_experiment.png" alt = "Higgs into fermions: Evidence of the Higgs boson decaying to fermions" width = "600">
    <figcaption> Fig 1. Higgs into fermions: Evidence of the Higgs boson decaying to fermions (image credit: CERN) </figcaption>
</figure>

## LHC at Work

**Proton-proton collisions.** In particle physics, an *event* refers to the results just after a [**fundamental interaction**](https://en.wikipedia.org/wiki/Fundamental_interaction) took place between subatomic particles, occurring in a very short time span, at a well-localized region of space. In the **LHC**, swarms of protons are accelerated on a circular trajectory in both directions, at an extremely high speed. These swarms are made to cross in the **ATLAS** detector, causing hundreds of millions of proton-proton collisions per second. The resulting **events** are detected by sensors, producing a sparse vector of about a hundred thousand dimensions (roughly corresponding to an image or speech signal in classical machine learning applications). The feature construction phase involves extracting type, energy, as well as $3$-D direction of each particle from the raw data. Also, the variable-length list of four-tuples is digested into a fixed-length vector of features containing up to tens of real-valued variables.

**Background events, signal events and selection region.** Some of these variables are first used in a real-time multi-stage cascade classifier (called the trigger) to discard most of the uninteresting events (called **background events**). The selected events (roughly four hundred per second) are then written on disks by a large CPU farm, producing petabytes of data per year. The saved events still, in large majority, represent known processes (these are also *background events*). The background events are mostly produced by the decay of particles which, though exotic in nature, are known beforehand from previous generations of experiments. The goal of the offline analysis is to find a region (called **selection region**) in the feature space that produces significantly excess of events (called **signal events**) compared to what known background processes can explain. Once the region has been fixed, a statistical test is applied to determine the significance of the excess. If the probability that the excess has been produced by background processes falls below a certain limit, it indicates the discovery of a new particle.

**The classification problem.** To optimize the selection region, multivariate classification techniques are routinely utilized. The formal objective function is unique and somewhat different from the classification error or other objectives that are used regularly in machine learning. Nevertheless, finding a *pure* signal region corresponds roughly to separating background events and signal events, which is a standard classification problem. Consequently, established classification methods are useful, as they provide better discovery sensitivity than traditional, manual techniques.

**Weighting and normalization.** The classifier is trained on simulated background events and signal events. Simulators produce weights for each event to correct for the mismatch between the prior probability of the event and the instrumental probability applied by the simulator. The weights are normalized such that in any region, the sum of the weights of events falling in the region gives an unbiased estimate of the expected number of events found there for a fixed integrated luminosity, which corresponds to a fixed data taking time for a given beam intensity. In this case, it corresponds to the data collected by the **ATLAS** experiment in $2012$. Since the probability of a signal event is usually several orders of magnitudes lower than the probability of a background event, the signal samples and the background samples are usually renormalized to produce a balanced classification problem. A real-valued discriminant function is then trained on this reweighted sample to minimize the weighted classification error. The signal region is then defined by cutting the discriminant value at a certain threshold, which is optimized on a held-out set to maximize the sensitivity of the statistical test.

**The broad goal is to improve the procedure that produces the selection region, i.e. the region (not necessarily connected) in the feature space which produces signal events.**

## Enter ML

**Shallow neural network.** Machine learning plays a major role in processing data resulting from experiments at particle colliders. The ML classifiers learn to distinguish between different types of collision events by training on simulated data from sophisticated Monte-Carlo programs. Shallow [**neural networks**](https://en.wikipedia.org/wiki/Neural_network) with single hidden layer are one of the primary techniques used for this analysis and standardized implementations are included in the prevalent multivariate analysis software tools used by physicists. Efforts to increase statistical power tend to focus on developing new features for use with the existing machine learning classifiers. These high-level features are non-linear functions of the low-level measurements, derived using knowledge of the underlying physical processes.

**Deep neural network.** The abundance of labeled simulation training data and the complex underlying structure make this an ideal application for [**deep learning**](https://en.wikipedia.org/wiki/Deep_learning), in particular for large, [**deep neural networks**](https://en.wikipedia.org/wiki/Deep_learning#Deep_neural_networks). Deep neural networks can simplify and improve the analysis of high-energy physics data by automatically learning high-level features from the data. In particular, they increase the statistical power of the analysis even without the help of manually derived high-level features.

## Data

**Source:** **https://www.kaggle.com/competitions/higgs-boson/data**

**The simulator.** The dataset has been built from official **ATLAS** full-detector simulation. The simulator has two parts. In the first, random proton-proton collisions are simulated based on the knowledge that we have accumulated on particle physics. It reproduces the random microscopic explosions resulting from the proton-proton collisions. In the second part, the resulting particles are tracked through a virtual model of the detector. The process yields simulated events with properties that mimic the statistical properties of the real events with additional information on what has happened during the collision, before particles are measured in the detector.

**Signal sample and background sample.** The signal sample contains events in which Higgs bosons (with a fixed mass of $125$ [**GeV**](https://en.wikipedia.org/wiki/Electronvolt)) were produced. The background sample was generated by other known processes that can produce events with at least one electron or muon and a hadronic tau, mimicking the signal. Only three background processes were retained for the dataset. The first comes from the decay of the $Z$ boson (with a mass of $91.2$ GeV) into two taus. This decay produces events with a topology very similar to that produced by the decay of a Higgs. The second set contains events with a pair of top quarks, which can have a lepton and a hadronic tau among their decay. The third set involves the decay of the $W$ boson, where one electron or muon and a hadronic tau can appear simultaneously only through imperfections of the particle identification procedure.

**Training set and test set.** The training set and the test set respectively contains $250000$ and $550000$ observations. The two sets share $31$ common features between them. Additionally, the training set contains **labels** (**signal** or **background**) and **weights**.

## Project Objective

**The objective of the project is to classify an event produced in the particle accelerator as background or signal**. As described earlier, a **background event** is explained by the existing theories and previous observations. A **signal event**, however, indicates a process that cannot be described by previous observations and leads to the potential discovery of a new particle.

## Evaluation Metric

The [**evaluation metric**](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers), used in this project, is the approximate median significance (AMS), given by

$$ AMS := \sqrt{2\left(\left(s+b+b_r\right)\log{\left(1+\frac{s}{b+b_r}\right)}-s\right)},$$

where
- $s:$ unnormalized [**true positive rate**](https://en.wikipedia.org/wiki/Sensitivity_and_specificity),
- $b:$ unnormalized [**false positive rate**](https://en.wikipedia.org/wiki/False_positive_rate),
- $b_r = 10:$ constant regularization term,
- $\log:$ [**natural logarithm**](https://en.wikipedia.org/wiki/Natural_logarithm).

Precisely, let $(y_1, \ldots, y_n) \in \left\\{\text{b},\text{s}\right\\}^n$ be the vector of true test labels (where $\text{b}$ indicates background event and $\text{s}$ indicates signal event) and let $(\hat{y}_1, \ldots, \hat{y}_n) \in \\{\text{b},\text{s}\\}^n$ be the vector of predicted test labels. Also let $(w_1, \ldots, w_n) \in {\mathbb{R}^+}^n$ be the vector of weights (where $\mathbb{R}^+$ denotes the set of positive real numbers). Then

$$ s = \sum_{i=1}^n w_i \mathbb{1}\left\\{y_i = s\right\\} \mathbb{1}\left\\{\hat{y_i} = s\right\\} $$

and

$$ b = \sum_{i=1}^n w_i \mathbb{1}\left\\{y_i = b\right\\} \mathbb{1}\left\\{\hat{y_i} = s\right\\}, $$

where the [**indicator function**](https://en.wikipedia.org/wiki/Indicator_function) $\mathbb{1}\left\\{S\right\\}$ is $1$ if $S$ is true and $0$ otherwise.