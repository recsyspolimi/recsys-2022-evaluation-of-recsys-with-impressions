# Recommender Systems Framework for Python 3.8

## Installation

Note that this repository requires Python 3.8

First we suggest you create an environment for this project using virtualenv (or another tool like conda)

First checkout this repository, then enter in the repository folder and run this commands to create and activate a new environment, if you are using conda:
```console
conda create -n RecSysFramework python=3.8 anaconda
conda activate RecSysFramework
```

In order to compile you must have installed: _gcc_ and _python3 dev_, which can be installed with the following commands:
```console
sudo apt install gcc 
sudo apt-get install python3-dev
```

Then install all the requirements and dependencies
```console
pip install -r requirements.txt
```


At this point you can compile all Cython algorithms by running the following command. The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. During the compilation you may see some warnings. 
 
```console
python run_compile_all_cython.py
```



## Project structure

### Base
Contains some basic modules and the base classes for different Recommender types.

#### Base.Evaluation
The Evaluator class is used to evaluate a recommender object. It computes various metrics:
* Accuracy metrics: ROC_AUC, PRECISION, RECALL, MAP, MRR, NDCG, F1, HIT_RATE, ARHR
* Beyond-accuracy metrics: NOVELTY, DIVERSITY, COVERAGE

The evaluator takes as input the URM against which you want to test the recommender, then a list of cutoff values (e.g., 5, 20) and, if necessary, an object to compute diversity.
The function evaluateRecommender will take as input only the recommender object you want to evaluate and return both a dictionary in the form {cutoff: results}, where results is {metric: value} and a well-formatted printable string.

```python

    from Evaluation.Evaluator import EvaluatorHoldout

    evaluator_test = EvaluatorHoldout(URM_test, [5, 20])

    results_run_dict, results_run_string = evaluator_test.evaluateRecommender(recommender_instance)

    print(results_run_string)

```


#### Base.Similarity
The similarity module allows to compute the item-item or user-user similarity.
It is used by calling the Compute_Similarity class and passing which is the desired similarity and the sparse matrix you wish to use.

It is able to compute the following similarities: Cosine, Adjusted Cosine, Jaccard, Tanimoto, Pearson and Euclidean (linear and exponential)

```python

    similarity = Compute_Similarity(URM_train, shrink=shrink, topK=topK, normalize=normalize, similarity = "cosine")

    W_sparse = similarity.compute_similarity()

```


### Recommenders
All recommenders inherit from BaseRecommender, therefore have the same interface.
You must provide the data when instantiating the recommender and then call the _fit_ function to build the corresponding model.

Each recommender has a _compute_item_score function which, given an array of user_id, computes the prediction or _score_ for all items.
Further operations like removing seen items and computing the recommendation list of the desired length are done by the _recommend_ function of BaseRecommender

As an example:

```python
    user_id = 158
    
    recommender_instance = ItemKNNCFRecommender(URM_train)
    recommender_instance.fit(topK=150)
    recommended_items = recommender_instance.recommend(user_id, cutoff = 20, remove_seen_flag=True)
    
    recommender_instance = SLIM_ElasticNet(URM_train)
    recommender_instance.fit(topK=150, l1_ratio=0.1, alpha = 1.0)
    recommended_items = recommender_instance.recommend(user_id, cutoff = 20, remove_seen_flag=True)
```

### Data Reader and splitter
DataReader objects read the dataset from its original file and save it as a sparse matrix.

DataSplitter objects take as input a DataReader and split the corresponding dataset in the chosen way.
At each step the data is automatically saved in a folder, though it is possible to prevent this by setting _save_folder_path = False_ when calling _load_data_.
If a DataReader or DataSplitter is called for a dataset which was already processed, the saved data is loaded.

DataPostprocessing can also be applied between the dataReader and the dataSplitter and nested in one another.

When you have bilt the desired combination of dataset/preprocessing/split, get the data calling _load_data_.

```python
dataset = Movielens1MReader()

dataset = DataPostprocessing_K_Cores(dataset, k_cores_value=25)
dataset = DataPostprocessing_User_sample(dataset, user_quota=0.3)
dataset = DataPostprocessing_Implicit_URM(dataset)

dataSplitter = DataSplitter_Warm_k_fold(dataset)

dataSplitter.load_data()

URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
```


#### Github Tokens
Run

git config --global credential.helper store

The first time it will ask you to login ad then will memorize the credentials

