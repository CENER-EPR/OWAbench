OWA Wake Modeling Challenge
-------------------------------------
`Javier Sanz Rodrigo <mailto:jsrodrigo@cener.com>`_, `Pawel Gancarski <mailto:pgancarski@cener.com>`_, `Fernando Borbón Guillén <mailto:fborbon@cener.com>`_, `Pedro Miguel Fernandes Correia <mailto:pmferandez@cener.com>`_


Background 
=========================
The `OWA Wake Modeling Challenge <https://www.carbontrust.com/media/677495/owa-wake-modelling-challenge_final-feb27.pdf>`_ is an Offshore Wind Accelerator (OWA) project that aims to improve confidence in wake models in the prediction of array efficiency. A benchmarking process comprising 5 wind farms allows wake model developers and end-users test their array efficiency prediction methodologies over a wide range of wind climate and wind farm layout conditions.

The project is integrated in the `IEA Task 31 Wakebench <https://community.ieawind.org/task31/home>`_ international framework for wind farm modeling and evaluation.

Scope and Objectives
====================
The `Anholt benchmark <https://thewindvaneblog.com/the-owa-anholt-array-efficiency-benchmark-436fc538597d>`_ has been a pilot to define, together with the participants, an open-source model evaluation methodology for array efficiency prediction which is available in this github repository. 

The challenge has been extended to `5 offshore wind farms <https://thewindvaneblog.com/owa-wake-modelling-challenge-extended-to-6-offshore-wind-farms-c76d1ae645c2>`_ to perform a multi-site assessment. The benchmarks are set up as a blind test so you won’t be able to access observational data. Instead, mesoscale simulation data is available for the modeller to decide on the best interpretation of the input data for the specific needs of the wake model. 

Benchmark participants are encouraged to test test the evaluation scripts of this repository, based on Jupyter notebooks, on own simulation data and submit their best predictions. It is also the best way to guarantee that the output data is submitted in the right format for post-processing. 

Data
====================
Input and annonymized output data is available from the benchmark shared folder:  
https://b2drop.eudat.eu/s/6C8CgyqZe3E3btq 

You can upload your formatted simulation data in the following "upload only" folders:

* **Anholt**: https://b2drop.eudat.eu/s/5AEL5XdNR6NqSDZ.
* **Dudgeon**: https://b2drop.eudat.eu/s/oJkR6x7nrjxomPN.
* **Ormonde**: https://b2drop.eudat.eu/s/XsfJS7sB2YMrDLG.
* **Rodsand2**: https://b2drop.eudat.eu/s/mGgmfdgYZw9aR92.
* **Westermost Rough**: https://b2drop.eudat.eu/s/sZqxDbXkA4FTG9H.

Your submissions will be checked for format compliance before adding them to the corresponding shared output folder. 

Citation
========
You can cite the github repo in the following way:

OWAbench. Version 2.0 (2019) https://github.com/CENER-EPR/OWAbench

Installation
============
We use Jupyter notebooks based on Python 3. We recomend the `Anaconda distribution <https://www.anaconda.com/distribution/>`_ to install python.

License
=======

Copyright 2019 CENER

Licensed under TBD