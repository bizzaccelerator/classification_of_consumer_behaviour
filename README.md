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

The proposed engineering solution is based on an `Optimized Gradient Boosted Tree model`, achieving an average deviation of 41.775 units from the test values and explaining 90.14% of the variability in corn yield production. This model outperformed other algorithms tested.

The model was selected after an extensive Exploratory Data Analysis (EDA), which addressed missing values, analyzed univariate distributions, and assessed feature importance. Details of the EDA process are available in the [notebook]().

The solution is implemented as a Python-based predictive service designed to estimate corn yields using survey data from farmers. It is deployed as a web application, enabling office teams to process survey data and predict expected corn yields for the current season, so they can take actions to reduce food insecurity in the county.

![Solution]()
Photo: Diagram of the solution engineered.

### _Rationale behind the solution:_ 

During the process different algorithms were tested. The first group analyzed were the Linear, Ridge and Lasso Regression; the second group studied were the Random Forest and it's the optimized version, and finally, the Gradient Boosted Trees and its Optimized version were taken into account too. An Optimized Gradient Boosted Tree model was chosen after evaluating various algorithms for its superior performance in balancing prediction accuracy and interpretability. 

The data used in this project was obtained for free from kagle [here](https://www.kaggle.com/datasets/japondo/corn-farming-data). However, a copy of the referred data is added to this repository for convenience. 

The application was coded in python using a distribution of Anaconda. Conda was used to manage isolated virtual environments and install all the packages needed without conflicts. This solution was built using Flask, a lightweight and flexible Python web framework, to efficiently handle HTTP requests and deliver a user-friendly interface for interacting with the predictive model. Flask was chosen for its simplicity, scalability, and suitability for developing APIs that serve the predictive service.

To ensure portability and consistent deployment across different environments, the application is containerized using Docker. This approach encapsulates the entire application through a [Dockerfile](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Dockerfile), including the Flask backend, the predictive model, and all dependencies, within a lightweight container. Docker allows the service to run seamlessly across various platforms, eliminating compatibility issues and simplifying deployment.

Together, Flask and Docker provide a robust foundation for the application [predict.py](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/predict.py), enabling efficient development, deployment, and scalability while ensuring reliability and ease of maintenance. 
