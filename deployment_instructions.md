# How to reproduce the project.

Follow the steps below to reproduce the project locally or on a remote environment:

### _1. Prerequisites:_
Ensure the following are installed on your system:

- Git - Download Git
- Docker - Install Docker
- (Optional) Python - If you'd like to run the application without Docker.

### _2. Clone the Repository:_

- Open a terminal and navigate to the desired folder.
- Clone the repository:

> git clone https://github.com/bizzaccelerator/classification_of_consumer_behaviour.git cd your-repo-name

### _3. Build the Docker Image:_

- Ensure Docker is running.
- Build the Docker image:

> docker build -t classification-users:latest .

## To test the service locally.

### _4.1 Run the Application:_

- Start a container:

> docker run -d -p 9696:9696 --name classification-app classification-user:latest

### _5.1 Access the application:_

- Open your web browser and go to: http://localhost:9696
- Alternatively, test with curl or Postman.

### _6.1 Testing the application:_

Open and run the file `test.ipynb`to get the classification predicted from the application. Please make sure the variable `customer` is updated as interested.

The variable farmer represents the information collected from surveys transmitted through the HTTP protocol, using JSON, as follows:

```
customer = {
        "user_id": 4,
        "device_model": "Google Pixel 5",
        "operating_system": 'Android',
        "app_usage_time_(min/day)": 239,
        "screen_on_time_(hours/day)": 4.8,
        "battery_drain_(mah/day)": 1676,
        "number_of_apps_installed": 56,
        "data_usage_(mb/day)": 871,
        "age": 20,
        "gender": "Male",
        "screen_on_time_(min/day)": 288.0}
```

### _7.1 Stopping and Removing the Container:_

- To stop the container:

> docker stop classification-app

- To remove the container:

> docker rm classification-app

## To deploy it to the cloud.

### _4.1 Run the Application:_