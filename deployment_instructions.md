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

### _4.2 Install the Prerequisites:_

- Create a free AWS account as indicated in this [article](https://mlbookcamp.com/article/aws).

- Ensure the AWS CLI is installed and configured on your local machine. If not installed follow this [instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html#install-msi) 

- Run aws configure command to set up your AWS details: 

> aws configure

> Provide: 

```
AWS Access Key ID: Your access key.
AWS Secret Access Key: Your secret key.
Default region name: E.g., us-east-1 (or any applicable region).
Default output format: Typically json, text, or table.
```

### _5.2 Push Your Docker Image to Amazon Elastic Container Registry (ECR):_

- Create an ECR Repository:

> aws ecr create-repository --repository-name your-repo-name

- Authenticate Docker with ECR:

> aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com

- Tag Your Docker Image:

> docker tag your-image-name:tag <account_id>.dkr.ecr.<region>.amazonaws.com/your-repo-name:tag

- Push Your Docker Image to ECR:

> docker push <account_id>.dkr.ecr.<region>.amazonaws.com/your-repo-name:tag

### _6.2 Define ECS Resources:_

- Create an ECS Cluster:

> aws ecs create-cluster --cluster-name your-cluster-name

- Create a file `task-definition.json` that contains:

```
{
  "family": "your-task-family",
  "containerDefinitions": [
    {
      "name": "your-container-name",
      "image": "<account_id>.dkr.ecr.<region>.amazonaws.com/your-repo-name:tag",
      "memory": 512,
      "cpu": 256,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 80,
          "hostPort": 80
        }
      ]
    }
  ]
}
```

- Register the Task Definition file as:

> aws ecs register-task-definition --cli-input-json file://task-definition.json

### _7.2 Create a Security Group:_

A security group acts as a virtual firewall to control inbound and outbound traffic to your ECS resources.

- Navigate to the EC2 Console:

```
Go to the AWS Management Console.
Open the EC2 dashboard.
```

- Create a Security Group:

```
Click on Security Groups under the "Network & Security" section.
Click Create Security Group.
Provide: 
Name: e.g., ecs-security-group
Description: Allow traffic for ECS service
VPC: Select your default VPC or the one you’re using.
```

- Add Inbound Rules:

```
Type            HTTP	
Protocol	TCP	
Port Range	80	
Source          0.0.0.0/0	
```

- Add rules to allow incoming traffic:

```
Outbound rules are typically open by default. Leave it as All traffic unless specific restrictions are needed.
```

- Save the Security Group.

