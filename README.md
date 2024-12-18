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