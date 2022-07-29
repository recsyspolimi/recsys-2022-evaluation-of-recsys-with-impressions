# Evaluation of Recommender Systems with Impressions

This repository contains the source code of the paper:

- Towards the Evaluation of Recommender Systems with Impressions. Fernando B. Pérez Maurera, Maurizio Ferrari Dacrema, and Paolo Cremonesi.  RecSys 2022. DOI: [10.1145/3523227.3551483](https://doi.org/10.1145/3523227.3551483).
  - [Paper Published Version (ACM DL)](https://doi.org/10.1145/3523227.3551483),
  - [Github](https://github.com/recsyspolimi/recsys-2022-evaluation-of-recsys-with-impressions)
  - If you use our work, please cite it. You can click on the Cite this repository button or use the following bibtex: 
    ```bibtex
    @inproceedings{conf/recsys/PerezFC22/towards-the-evaluation-of-recommender-systems-with-impressions,
      author    = {P{\'{e}}rez Maurera, {Fernando B.} and {Ferrari Dacrema}, Maurizio and Cremonesi, Paolo},
      title     = {Towards the Evaluation of Recommender Systems with Impressions},
      booktitle = {Sixteenth ACM Conference on Recommender Systems (RecSys '22), September 18--23, 2022, Seattle, WA, USA},
      url       = {https://doi.org/10.1145/3523227.3551483},
      doi       = {10.1145/3523227.3551483},
    }
    ```
See our [website](http://recsys.deib.polimi.it/) for more information on our research group. We are actively pursuing
this research direction in evaluation and reproducibility, we are open to collaboration with other researchers. Follow
our project on [ResearchGate](https://www.researchgate.net/project/Recommender-systems-reproducibility-and-evaluation)

This repo is divided in three folders:
- [impressions-evaluation](impressions-evaluation/README.md): Contains the implementation of the impressions 
  recommenders (time-aware, re-ranking, impressions as user profiles), scripts to create the dataset splits and run the experiments. 
- [recsys-framework](RecSysFramework_public/README.md): Contains the implementation of baselines, hyper-parameter 
  tuning loop, and evaluation framework.
- [recsys-framework-extensions](recsys-framework-extensions): Contains utility classes and functions that extend the 
  functionalities provided by the `recsys-framework`. Functions for common data-splitting are defined here.

You'll find instructions to install this project and run the experiments in the  
[README inside impressions-evaluation](impressions-evaluation/README.md).

NOTE: all commands must be run from a terminal using `poetry` and inside the `impressions-evaluation` folder.

## Contact us
The best place to ask questions or raise issues is in the [issues' page](https://github.com/recsyspolimi/recsys-2022-evaluation-of-recsys-with-impressions/issues)
of the repo. For further inquiries, you may contact [Fernando Pérez via email](mailto:fernandobenjamin.perez@polimi.it).
