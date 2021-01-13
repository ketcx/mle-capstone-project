<h1 align='center'>Predict whether hotel booking will be canceled</h1>
<p align="center">Armando Medina</p>
<p align="center">(January, 2021)</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/diagram.png" width=600>
</p>

### Project Overview

<p>In this project we are going to work with the data set of the reservations of two hotels located in the city of Lisbon, Portugal. We are going to use Azure Machine Learning to set up a cloud-based machine learning production model, deploy it, and consume it. But first we are going to compare the result between a logistic regression model with the parameters tunning through Hypedrive and applying AutoML to the same dataset.</p>

<p>The objective is to have a web service that can predict when a hotel reservation will be canceled or not in order to have better planning.</p>

<br />
<p>Main steps of the project:

1. Dataset
2. Overview
3. Deploy the best model
4. Task
5. Access
6. Automated ML
7. Results
8. Hyperparameter Tuning
9. Results
10. Model Deployment
11. Screen Recording
12. OStandout Suggestions
</p>
<br />
<hr />
<br />

### Dataset

<p>
For this project we are going to use a dataset obtained from Kaggle that collects information on reservations from two hotels in Portugal. 
</p>

[Link](https://www.kaggle.com/jessemostipak/hotel-booking-demand)


<p>
This dataset was recompiled with the objective of developing prediction models to classify the probability of cancellation of a hotel reservation. However, due to the characteristics of the variables included in these data sets, their use goes beyond this cancellation prediction problem.
</p>
<p>
The data is originally from the article Hotel Booking Demand Datasets, written by Nuno Antonio, Ana Almeida, and Luis Nunes for Data in Brief, Volume 22, February 2019. 
<p>

[Link](https://www.sciencedirect.com/science/article/pii/S2352340918315191)

</p>

### Task
I decided to do the cleaning operations on the notebook: hotel-boking-demand-dataset-cleanup.ipynb. In that nootebook I removed the notebook values and unnecessary columns.

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img003.png" width=600>
</p>

### Access
After that I added the clean data set through the Azure Machine Learning interface to be able to consume it within my workspace.

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img001.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img002.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img004.png" width=600>
</p>

### Automated ML
For the AutoML settings and configuration, I chose as a metric to evaluate the precision, I put a timeout of 30 minutes in the experiment, I enabled the early stop policy, I limited the time of each interaction to 15 minutes and I put a maximum of 5 concurrent nodes .

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
<p>
The best general model was the VotingEnsemble, which is an ensemble machine learning model that combines the predictions of all the previous models, but taking this the best model was the MaxAbsScaler, LightGBM with an accuracy of 87% and a duration of 01:26 seconds of training. , while the worst model was the StandardScalerWrapper, LightGBM with 62% accuracy and a duration of 01:25, which indicates that with a similar training time the MaxAbsScaler, LightGBM achieved an accuracy of 25% better using the same model but a different data processing technique, MaxAbsScaler instead of StandardScalerWrapper.
</p>
<p>
The parameters used for the AutoMl were:

* n_estimators=100
* n_jobs=1
* nthread=None
* objective='reg:logistic'
*  random_state=0
* reg_alpha=0
*  reg_lambda=2.0833333333333335
* scale_pos_weight=1
* seed=None
* silent=None
* subsample=1
* tree_method='auto'
* verbose=-10
* verbosity=0
</p>

<p>In conclusion, due to the result I think that for future work I think that the performance can be improved by working a little more in the configuration of the featurization.</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img005.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img006.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img007.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img008.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img009.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img010.png" width=600>
</p>

### Hyperparameter Tuning

<p>
For my manual model as a preprocessing of my data I used LabelEncoder on the string type columns and then I chose two hyperparameters:

* max_iter: Maximum number of iterations of the optimization algorithm.

* C: Each of the values in C describes the inverse of regularization strength. Like in support vector machines, smaller values specify stronger regularization.

As samplin's method I chose RandomSampling is motivated because Regularization Strength is a continuous hyperparameter. In other words, random sampling allowed my parameters to be initialized with both discrete and continuous values, and it also allowed for early political termination. This choice gave us an appropriate cost / benefit result.

Just add that for the maximum number of iterations we use the choice method with discrete hyperparameters of 100, 1000, 10000 as options and for C we use continuous values in the range of 0.01 to 100, hence we use the uniform method.
</p>

### Results

After running our experiment with four interactions, our best result was:

* Regularization Strength:: 62.88105217581512
* Max iterations:': 10000, '
* Accuracy': 0.796409697173056

<p>For future work we can use the processing used by AutoML in the best MaxAbsScaler model. Also looking at how the precision was behaving, you could try adjusting other hyperparameters in addition to changing the lower range of C to start from a larger value, as with these results, use the Grind Sampling method.</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img011.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img012.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img013.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img014.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img015.png" width=600>
</p>

### Comparing Models

Comparing the model with the tuning parameters with HyperDrive with AutoML the results are:

- For the best AutoML model:

  - 93% of the "no" were classified correctly.
  - 7% of the "no" were classified as "yes" incorrectly.
  - 78% of the "yes" were classified correctly.
  - 22% of the "yes" were classified as "no" incorrectly.

- For the best model with hyperparameters optimized with HyperDrive:

  - 92% of the "no" were classified correctly.
  - 8% of the "no" were classified as "yes" incorrectly.
  - 58% of the "yes" were classified correctly.
  - 42% of the "yes" were classified as "no" incorrectly.

<p>Observing the results we can conclude that the best model of applying AutoML predicts with better precision the reservations that can be canceled, 78% against 58% for the best model with hyperparameters optimized with HyperDrive
### Model Deployment</p>

*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

Comparing the precision results the model that you select to deploy the best AutoML model, (although also deploy to test the option of "No code deployment").

<b>Sample input:</b><br />

```python

data = {
    "data": [
     {
          "hotel": "Resort Hotel",
          "lead_time": 85,
          "stays_in_weekend_nights": 0,
          "stays_in_week_nights": 3,
          "adults": 2,
          "children": 0,
          "babies": 0,
          "meal": "BB",
          "country": "PRT",
          "market_segment": "Online TA",
          "distribution_channel": "TA/TO",
          "is_repeated_guest": 0,
          "booking_changes": 0,
          "deposit_type": "No Deposit",
          "agent": 240,
          "company": 0,
          "days_in_waiting_list": 0,
          "customer_type": "Trasient",
          "adr": 82,
          "required_car_parking_spaces": 1,
          "total_of_special_requests": 1,
          "room": 1,
          "net_cancelled": 0
     }
   ]
}
```

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img016.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img017.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img018.png" width=600>
</p>



### Screen Recording

[![Screencast video](https://img.youtube.com/vi/Atlm5DUkH6M/0.jpg)](https://www.youtube.com/watch?v=Atlm5DUkH6M)

### Standout Suggestions
I implemented two prominent suggestions: Convert your model to ONNX format & Enable logging.

<p><strong>Convert your model to ONNX format.</strong> Open Neural Network Exchange (ONNX) can help optimize the inference of your machine learning model. Inference, or model scoring, is the phase where the deployed model is used for prediction, most commonly on production data.

For this reason, enable the ONNX format support option for the model in the AutoML configuration and then save the model with the help of OnnxConverter.</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img019.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img020.png" width=600>
</p>

<p><strong>Enable logging in your deployed webservices.</strong> When deploying my model enable the "enable_app_insights" option which is disabled by default. This enabled the App insights in my webservices which allowed me to record useful data about the requests that were sent, among which were: the time of the inference, the time the request arrived and other details.</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img021.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img022.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/img023.png" width=600>
</p>
