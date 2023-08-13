# Higgs Boson Event Detection Project


 
![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Microsoft Excel](https://img.shields.io/badge/Microsoft_Excel-217346?style=for-the-badge&logo=microsoft-excel&logoColor=white)
![Canva](https://img.shields.io/badge/Canva-%2300C4CC.svg?style=for-the-badge&logo=Canva&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Markdown](https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor=white)
![Microsoft Office](https://img.shields.io/badge/Microsoft_Office-D83B01?style=for-the-badge&logo=microsoft-office&logoColor=white)
![Microsoft Word](https://img.shields.io/badge/Microsoft_Word-2B579A?style=for-the-badge&logo=microsoft-word&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Windows Terminal](https://img.shields.io/badge/Windows%20Terminal-%234D4D4D.svg?style=for-the-badge&logo=windows-terminal&logoColor=white)

**Particle accelerators.** Physicists use particle accelerators to study the fundamental aspects of matter, space, and time by observing interactions between subatomic particles during high-energy collisions. These collisions provide insights into the structure of these elements. However, the experimental measurements from these collisions are imprecise. To enhance the analysis of this data, researchers utilize machine learning (ML). They employ standardized ML software packages to extract significant features from the raw measurements, aiming to improve statistical accuracy and enhance their understanding of these complex phenomena.

**Higgs boson.** The Higgs boson, sometimes called the God particle, is a crucial component of the standard model of particle physics. This model outlines the fundamental particles and forces that make up the universe. While elementary particles are expected to be without mass at high energies, some gain mass at lower energies, a mystery in theoretical physics for a long time. In 1964, Peter Higgs and others proposed a mechanism explaining the origin of particle mass. This involves a field known as the Higgs field, which particles interact with to acquire mass; the more interaction, the greater the mass. Some particles, like photons, don't interact with this field and remain massless. The Higgs boson is tied to the Higgs field and represents its physical form, granting mass to other particles. Detecting this elusive particle took nearly fifty years since its theoretical proposal.

**The discovery.** On July 4th, 2012, the ATLAS and CMS experiments conducted at CERN's Large Hadron Collider reported the detection of a new particle with a mass around 125 GeV. This particle aligns with the theoretical Higgs boson. The discovery led to François Englert and Peter Higgs being awarded the Nobel Prize in Physics in 2013. Their recognition stemmed from their theoretical work on a mechanism that enhances our comprehension of subatomic particle mass origins. This discovery was subsequently verified by the ATLAS and CMS experiments at CERN's Large Hadron Collider.

**Giving mass to fermions.** The Higgs boson can decay into various particles, with these decay pathways referred to as "channels" in physics. It was initially observed to decay into three distinct channels involving boson pairs. To prove that the Higgs field gives mass to fermions (particles like quarks and leptons), it needs to be shown that the Higgs boson can directly decay into fermion pairs. Investigating the decay of the Higgs boson into fermion pairs, such as tau leptons or b-quarks, and accurately measuring their properties, has become an important research focus. Among the different decay pathways, the most promising one involves the decay into a pair of tau leptons, which offers a good balance between a reasonable branching ratio and manageable background noise.

**The first evidence of $h \to \tau^+\tau^-$ decays [was recently reported](https://cds.cern.ch/record/1632191), based on the full set of proton–proton collision data recorded by the ATLAS experiment at the LHC during $2011$-$2012$. Despite the consistency of the data with $h \to \tau^+\tau^-$ decays, it could not be ensured that the statistical power exceeds the $5\sigma$ threshold, which is the required standard for claims of discovery in high-energy physics community.**


## Project Structure

The project repository is organized as follows:

```

├── LICENSE
├── README.md           <- README .
├── notebooks           <- Folder containing the final reports/results of this project.
│   │
│   └── Higgs Boson Event Detection.py   <- Final notebook for the project.
├── reports            <- Folder containing the final reports/results of this project.
│   │
│   └── Report.pdf     <- Final analysis report in PDF.
│   
├── src                <- Source for this project.
│   │
│   └── data           <- Datasets used and collected for this project.
|   └── model          <- Model.

```

## LHC at Work

**Proton-proton collisions.** In particle physics, an "event" is what happens when tiny particles interact with each other. This happens really quickly and in a small space. In a big machine called the LHC, small particles are sent zooming around in circles really fast. They crash into each other in a special part of the machine called the ATLAS detector. This collision happens millions of times each second. Sensors in the detector notice these collisions and make a kind of list with lots of numbers, kind of like a picture or sound in regular computer learning. People then figure out important things about the particles from this list, like what type they are, how much energy they have, and which way they're moving. They also turn a special list of four sets of numbers into a shorter list with important information about the particles.

**Background events, signal events and selection region.** In a multi-stage process, certain variables are initially used in a real-time cascade classifier to filter out less relevant events referred to as "background events." The chosen events are then stored using a CPU farm, resulting in a significant amount of data. Most of these saved events are still related to known processes, also categorized as "background events." These background events mostly arise from the decay of particles, which are already familiar from past experiments. The main objective of the subsequent offline analysis is to identify a specific area in the feature space known as the "selection region." This region should exhibit a notably higher number of events called "signal events" compared to what established background processes can account for. Once this region is defined, a statistical test is conducted to determine the significance of the excess of events. If the likelihood of this excess being due to background processes falls below a certain threshold, it implies the discovery of a new particle.

**The classification problem.** To optimize the selection region, multivariate classification techniques are routinely utilized. The formal objective function is unique and somewhat different from the classification error or other objectives that are used regularly in machine learning. Nevertheless, finding a *pure* signal region corresponds roughly to separating background events and signal events, which is a standard classification problem. Consequently, established classification methods are useful, as they provide better discovery sensitivity than traditional, manual techniques.

**Weighting and normalization.** The classifier is trained on simulated background events and signal events. Simulators produce weights for each event to correct for the mismatch between the prior probability of the event and the instrumental probability applied by the simulator. The weights are normalized such that in any region, the sum of the weights of events falling in the region gives an unbiased estimate of the expected number of events found there for a fixed integrated luminosity, which corresponds to a fixed data taking time for a given beam intensity. In this case, it corresponds to the data collected by the **ATLAS** experiment in $2012$. Since the probability of a signal event is usually several orders of magnitudes lower than the probability of a background event, the signal samples and the background samples are usually renormalized to produce a balanced classification problem. A real-valued discriminant function is then trained on this reweighted sample to minimize the weighted classification error. The signal region is then defined by cutting the discriminant value at a certain threshold, which is optimized on a held-out set to maximize the sensitivity of the statistical test.

**The broad goal is to improve the procedure that produces the selection region, i.e. the region (not necessarily connected) in the feature space which produces signal events.**

## Approach

**Shallow neural network.** Machine learning plays a major role in processing data resulting from experiments at particle colliders. The ML classifiers learn to distinguish between different types of collision events by training on simulated data from sophisticated Monte-Carlo programs. Shallow [**neural networks**](https://en.wikipedia.org/wiki/Neural_network) with single hidden layer are one of the primary techniques used for this analysis and standardized implementations are included in the prevalent multivariate analysis software tools used by physicists. Efforts to increase statistical power tend to focus on developing new features for use with the existing machine learning classifiers. These high-level features are non-linear functions of the low-level measurements, derived using knowledge of the underlying physical processes.

**Deep neural network.** The abundance of labeled simulation training data and the complex underlying structure make this an ideal application for [**deep learning**](https://en.wikipedia.org/wiki/Deep_learning), in particular for large, [**deep neural networks**](https://en.wikipedia.org/wiki/Deep_learning#Deep_neural_networks). Deep neural networks can simplify and improve the analysis of high-energy physics data by automatically learning high-level features from the data. In particular, they increase the statistical power of the analysis even without the help of manually derived high-level features.

## Data

**Data Available:** **[Here](https://github.com/tushar2704/Higgs_Boson_Event_Detection/tree/main/src/data)**

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


## License

This project is licensed under the [MIT License](LICENSE).
## Author
- <ins><b>©2023 Tushar Aggarwal. All rights reserved</b></ins>
- <b>[LinkedIn](https://www.linkedin.com/in/tusharaggarwalinseec/)</b>
- <b>[Medium](https://medium.com/@tushar_aggarwal)</b> 
- <b>[Tushar-Aggarwal.com](https://www.tushar-aggarwal.com/)</b>
- <b>[New Kaggle](https://www.kaggle.com/tagg27)</b> 

## Contact me!
If you have any questions, suggestions, or just want to say hello, you can reach out to us at [Tushar Aggarwal](mailto:info@tushar-aggarwal.com). We would love to hear from you!

