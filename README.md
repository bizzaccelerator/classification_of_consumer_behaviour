# Classification of consumer behaviour for a Cellphone Original Equipment Manufacturer (OEM) startup company 

*A Python-based predictive application for segment customers into distinct classes according it's phone usage behaviour using survey data, containerized with Docker.*

This fictional project was developed as a Capstone project for the Machine Learning Zoomcamp offered by Data Talks Club. Method and objectives were defined for educational purposes only, so I can show the knowledge appropiated during the training. 

The current project simulates a real scenario of information gathering to help a marketing team of a startup company who pretends to lauch a novel cellphone to better understand and predict customer needs, enabling more targeted marketing and decision-making. 

![phone_usage](https://github.com/bizzaccelerator/classification_of_consumer_behaviour/blob/main/Images/phone_usage.jpg)
Photo: ©Julio Lopez – pexels.com

## Problem statement
This could be understood in two leves: a business problem and a technical problem. 

### _Business problem:_
The cellphone market is highly competitive, making it essential to understand consumer preferences to maintain a competitive edge. A cellphone OEM must identify behavioral patterns to predict customer actions, such as upgrade cycles, brand loyalty, and feature preferences. For a startup, this information is even more critical.

The marketing team of this cellphone OEM needs insights into customer usage patterns within a targeted segment and estimates of potential benefits. This knowledge enables them to tailor marketing efforts, refine product features, and boost customer satisfaction. Therefore, the marketing team requires a tool that categorizes the analyzed individuals into different classes, enabling appropriate actions to be taken. 

### _Technical problem:_
As a Machine Learning engineer, I am tasked with developing a classification model to predict customer segments based on behavioral and demographic data. This model will enable the marketing team to categorize present and future leads into five actionable classes: 'Cost-Conscious Customer,' 'Feature-Oriented Buyer,' 'Brand Loyalist,' and others. To achieve this goal, the model explores data collected from various cellphone users, identifying several useful variables such as gender, age, daily data usage, and screen time exposure, among others. This model is implemented in a cloud solution that serves the model for future use and insights extraction, enhancing its reliability, readability, and security.

## Solution proposed

The proposed engineering solution is based on an `Optimized Random Forest model` that accurately classifies all subjects in the test dataset, achieving 100% precision and recall. This model outperformed other algorithms tested.

The model was selected after an extensive Exploratory Data Analysis (EDA), which addressed missing values, analyzed univariate distributions, and assessed feature importance. Details of the EDA process are available in the [notebook](https://github.com/bizzaccelerator/classification_of_consumer_behaviour/blob/main/notebook.ipynb).

The solution is implemented as a Python-based predictive service that estimates a customer's class in advance based on survey data of usage patterns. It is deployed as a web application, enabling the marketing team of the OEM to process survey data and predict expected consumer behaviour, so they can target them with devices that best meet their needs.

![Solution](https://github.com/bizzaccelerator/classification_of_consumer_behaviour/blob/main/Images/Classify.jpg)
Photo: Diagram of the solution engineered.

### _Rationale behind the solution:_ 

During the process different algorithms were tested. The first group analyzed was Decision Trees, followed by Random Forests, Gradient Boosted Trees, Support Vector Classifiers, and finally, Convolutional Neural Networks (CNN). The best parameters for all these algorithms were identifyed using GridSearch. An Optimized Random Forest Model model was chosen after evaluating various algorithms for its superior performance in balancing prediction accuracy and interpretability. 

The data used in this project was obtained for free from kagle [here](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset). However, a copy of the referred data is added to this repository for convenience. 

The application was coded in python using a distribution of Anaconda. Conda was used to manage isolated virtual environments and install all the packages needed without conflicts. The [final_project_1.yml](https://github.com/bizzaccelerator/classification_of_consumer_behaviour/blob/main/final_project_1.yml) file contains the enviroment configuration needed using conda. For those who doesn't use conda for managing packages, feel free to use the [requirements.txt](https://github.com/bizzaccelerator/classification_of_consumer_behaviour/blob/main/requirements.txt) file to reproduce the project. 

The solution was built using Flask, a lightweight and flexible Python web framework, to efficiently handle HTTP requests and deliver a user-friendly interface for interacting with the predictive model. Flask was chosen for its simplicity, scalability, and suitability for developing APIs that serve the predictive service.

To ensure portability and consistent deployment across different environments, the application is containerized using Docker. This approach encapsulates the entire application through a [Dockerfile](https://github.com/bizzaccelerator/classification_of_consumer_behaviour/blob/main/Dockerfile), including the Flask backend, the predictive model, and all dependencies, within a lightweight container. Docker allows the service to run seamlessly across various platforms, eliminating compatibility issues and simplifying deployment.

Together, Flask and Docker provide a robust foundation for the application [predict.py](https://github.com/bizzaccelerator/classification_of_consumer_behaviour/blob/main/predict.py), enabling efficient development, deployment, and scalability while ensuring reliability and ease of maintenance. 

The engineered classification service was deployed using AWS Elastic Container Service (ECS), a powerful orchestration service designed to simplify the deployment and management of containerized applications in a reliable and scalable environment. ECS handles critical tasks such as container scheduling, resource management, and performance monitoring, enabling the team to concentrate on enhancing the accuracy and efficiency of the classification system. Its flexibility ensures the solution seamlessly adapts to growing user demands while maintaining consistent performance, making it an optimal choice for delivering actionable insights into consumer behavior based on consumption patterns. My application is running at `http://35.175.152.2/`*(The solution will be avaliable online until the project is evaluated at the end of the Machine Learning Zoomcamp by Data Talks Club)*.

## How to run the project.

Follow the this [instructions](https://github.com/bizzaccelerator/classification_of_consumer_behaviour/blob/main/deployment_instructions.md) to reproduce the project.