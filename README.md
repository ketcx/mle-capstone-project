<h1 align='center'>Predict whether hotel booking will be canceled</h1>
<p align="center">Armando Medina</p>
<p align="center">(January, 2021)</p>

<p align="center">
  <img src="https://github.com/ketcx/mle-capstone-project/blob/master/data/diagram.png" width=600>
</p>

## Project Overview

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
12. ONNX & Enable logging
13. Future Works
</p>
<br />
<hr />
<br />

## Dataset

<p>
For this project we are going to use a dataset obtained from Kaggle that collects information on reservations from two hotels in Portugal. [Link](https://www.kaggle.com/jessemostipak/hotel-booking-demand)
</p>
<p>
This dataset was recompiled with the objective of developing prediction models to classify the probability of cancellation of a hotel reservation. However, due to the characteristics of the variables included in these data sets, their use goes beyond this cancellation prediction problem.
</p>
<p>
The data is originally from the article Hotel Booking Demand Datasets, written by Nuno Antonio, Ana Almeida, and Luis Nunes for Data in Brief, Volume 22, February 2019. [Link](https://www.sciencedirect.com/science/article/pii/S2352340918315191)
<p>
</p>
<p>
The data was downloaded and cleaned by Thomas Mock and Antoine Bichat for TidyTuesday during the week of February 11th, 2020.
</p>

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.


## 1. Authentication

<p>In this step, I install Azure Machine Learning Extension which allows you to interact with Azure Machine Learning Studio, which is part of the az command.

After you have the Azure Machine Learning Extension, create a service principal account and then associate it with the specific workspace.</p>

<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img001.png" width=600>
</p>

## 2. Automated ML Experiment

<p>In this step, I will create an experiment using automated machine learning and then configure a compute cluster and use that cluster to run the experiment.</p>
<br />
<p align="center">Data set registered in Azure ML from a url.</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img002.png" width=600>
  
</p>
<br />
<p align="center">We apply AutoML to our dataset. Here is the completed experiment.</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img003.png" width=600>  
</p>
<br />
<p align="center">Here we see the best model of our experiment: MaxAbSacler, XGBoostClasssifier.</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img004.png" width=600>  
</p>

## 3. Deploy the best model

<p>In this step, implement the best model to be able to interact with the HTTP API service and interact with the model by sending data through POST requests.</p>

## 4. Enable logging

<p>Now that the best model has been implemented, enable Application Insights and retrieve the logs through a script.</p>

<p align="center">Here we see "Application Insights" enable inthe deatials tab of the endpoint.</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img005.png" width=600>  
</p>

<p align="center">Here we see the output when you run logs.py.</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img006.png" width=600>  
</p>

## 5. Swagger Documentation

<p>In this step, you will consume the deployed model using Swagger.</p>

<p align="center">Here we see swagger run on localhost showing the HTTP API methods and reponse for the model.</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img007.png" width=600>  
</p>

## 6. Consume model endpoints

<p>In this step I used the provided endpoint.py script to interact with the trained model.</p>

<p align="center">Here we see the output of endpoint.py.</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img008.png" width=600>  
</p>

<p align="center">Here we see the output of Apache Benchmark run against the HTTP API.</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img009.png" width=600>  
</p>

<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img010.png" width=600>  
</p>

## 7. Create and publish a pipeline

<p >
This step shows our work with:

- The pipeline section of Azure ML studio, showing that the pipeline has been created.
- The pipelines section in Azure ML Studio, showing the Pipeline Endpoint.
- The Bankmarketing dataset with the AutoML module
- The “Published Pipeline overview”, showing a REST endpoint and a status of ACTIVE.
- In Jupyter Notebook, showing that the “Use RunDetails Widget” shows the step runs.
- In ML studio showing the scheduled run.

</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img011.png" width=600>  
</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img012.png" width=600>  
</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img013.png" width=600>  
</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img014.png" width=600>  
</p>

<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img015.png" width=600>  
</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img016.png" width=600>  
</p>
<p align="center">
  <img src="https://github.com/ketcx/ml-ops-exercise/blob/master/data/img017.png" width=600>  
</p>

## 8. Screencast
  
[![Screencast video](https://img.youtube.com/vi/q3jSMlE9Q0c/0.jpg)](https://www.youtube.com/watch?v=q3jSMlE9Q0c)

## 9. Future Works

<p>In step some we create a service principal to be able to authenticate in this way, however in our script we authenticate ourselves with the help of the config.json file that we download from Azure Machine Learning, in future work in addition to better encapsulating some elements we can use the service princial as authentication method for our scripts.</p>
