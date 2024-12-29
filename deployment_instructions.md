## How to run the project.

Follow the steps below to reproduce the project locally or on a remote environment:

### _1. Prerequisites:_
Ensure the following are installed on your system:

- Git - Download Git
- Docker - Install Docker
- (Optional) Python - If you'd like to run the application without Docker.

### _2. Clone the Repository:_

- Open a terminal and navigate to the desired folder.
- Clone the repository:

> git clone https://github.com/bizzaccelerator/corn-yield-prediction.git cd your-repo-name

### _3. Build the Docker Image:_

- Ensure Docker is running.
- Build the Docker image:

> docker build -t corn-yield-prediction:latest .

### _4. Run the Application:_

- Start a container:

> docker run -d -p 9696:9696 --name corn-yield-app corn-yield-prediction:latest

### _5. Access the application:_

- Open your web browser and go to: http://localhost:9696
- Alternatively, test with curl or Postman.

### _6. Testing the application:_

Open and run the file `test.ipynb`to get the corn yield predicted from the application. Please make sure the variable `farmer` is updated as interested.

The variable farmer represents the information collected from surveys transmitted through the HTTP protocol, using JSON, as follows:

```
farmer = {"education": "Certificate",
        "gender": "Male",
        "age_bracket": "36-45",
        "household_size": "7",
        "acreage": "1.5",
        "fertilizer_amount": 300,
        "laborers": "3",
        "main_credit_source": "Credit groups",
        "farm_records": "Yes",
        "main_advisory_source": "Radio",
        "extension_provider": "County Government",
        "advisory_format": "SMS text",
        "advisory_language": "Kiswahili"}
```

### _7. Stopping and Removing the Container:_

- To stop the container:

> docker stop corn-yield-app

- To remove the container:

> docker rm corn-yield-app
