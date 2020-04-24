OWA Wake Modeling Challenge
-------------------------------------
`Javier Sanz Rodrigo <mailto:jsrodrigo@cener.com>`_, `Pawel Gancarski <mailto:pgancarski@cener.com>`_, `Pedro Miguel Fernandes Correia <mailto:pmferandez@cener.com>`_


Background 
=========================
The `Offshore Wind Accelerator <hhttps://www.carbontrust.com/es/node/981>`_ (OWA) Wake Modeling Challenge aims to improve confidence in wake models in the prediction of array efficiency. A benchmarking process comprising 5 wind farms allows wake model developers and end-users test their array efficiency prediction methodologies over a wide range of wind climate and wind farm layout conditions.

The project is integrated in the Phase 3 of the `IEA Task 31 Wakebench <https://community.ieawind.org/task31/home>`_ international framework for wind farm modeling and evaluation.

Scope and Objectives
====================
The primary objective of the project is to understand the limitations of wake models used by industry for the prediction of array efficiency over a relevant range of operational conditions in the offshore environment. To this end, the following three objectives were defined to guide the Wake Modelling Challenge:

* Evaluate wake modelling and power prediction methods and validate the results with measured data.
* Examine the accuracy of specific models, quantify uncertainty bands and highlight modelling trends.
* Define an open-access model evaluation methodology that can systematically improve consistency and traceability in the assessment of state-of-the-art wake models for power prediction as more datasets are added.

Benchmark Guides
================
The following blog posts were used to guide benchmark participants:
* `Anholt benchmark <https://thewindvaneblog.com/the-owa-anholt-array-efficiency-benchmark-436fc538597d>`_  
* `5 additional offshore wind farms <https://thewindvaneblog.com/owa-wake-modelling-challenge-extended-to-6-offshore-wind-farms-c76d1ae645c2>`_  

Data
====================
Benchmark input data and simulation data is published open-access in the following data repository: 

Citation
========
You can cite the github repo in the following way:

OWAbench. Version 2.0 (2020) https://github.com/CENER-EPR/OWAbench

Installation
============
We use Jupyter notebooks based on Python 3. We recomend the `Anaconda distribution <https://www.anaconda.com/distribution/>`_ to install python. 

Dependencies
============
The libraries used by the notebooks can be installed with 

```bash
pip install -r requirements.txt
```

License
=======
Copyright 2020 CENER
Licensed under the GNU General Public License v3.0

Acknowledgements
================
The authors would like to thank Carbon Trust and the OWA Technical Working Group for their support providing funding, operational data and guidance throughout the project. We would like to thank all the benchmark participants for their simulations and in-kind support in fine-tuning the benchmark set-up and evaluation methodology.
