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
# For docker connection
Type            HTTP	
Protocol	TCP	
Port Range	80	
Source          0.0.0.0/0	

# For ssh connection
Type            SSH	
Protocol	TCP	
Port Range	22	
Source          Your IP address	
```

- Add rules to allow incoming traffic:

```
Outbound rules are typically open by default. Leave it as All traffic unless specific restrictions are needed.
```

- Save the Security Group.

### _8.2 Create or Use an Existing VPC and Subnets:_

ECS services require a VPC and subnets for network communication.

- Create a VPC (if necessary):

```
Go to the VPC Console.
Click Create VPC.
Configure:      
        Name tag: e.g., ecs-vpc      
        IPv4 CIDR block: e.g., 10.0.0.0/16
```

- Create Subnets:

```
In the VPC console, create subnets for your VPC.
Use different availability zones for high availability.
Example CIDR blocks for subnets:
        10.0.1.0/24 (Subnet in us-east-1a)
        10.0.2.0/24 (Subnet in us-east-1b)
```
### _9.2 Configure an Internet Gateway:_

To allow public access to your application:

- Attach an Internet Gateway to the VPC:

```
In the VPC Console, create an Internet Gateway.
Attach it to your VPC.
```

- Route Table Setup:

```
Edit the VPC's route table.
Add a route:
        Destination: 0.0.0.0/0
        Target: The Internet Gateway you created.
```

### _10.2 Launch an ECS-Optimized EC2 Instance:_

- Go to the EC2 Console:

```
Open the EC2 Dashboard in the AWS Management Console.
```

- Launch a New Instance:

```
Click Launch Instances.
```

- Choose an Amazon Machine Image (AMI):

```
Select an ECS-Optimized AMI. For Amazon Linux 2
```

- Choose an Instance Type:

```
Select an appropriate instance type. For testing or low-traffic environments, t2.micro or t3.micro (Free Tier eligible) is a good choice.
```

- Configure Instance Details:

```
- Network: Choose the VPC you created or are using for ECS.
- Subnet: Select a public subnet associated with the route table and Internet Gateway.
- Auto-assign Public IP: Ensure this is enabled so the instance gets a public IP.
- IAM Role: Select or create an IAM role with the policy AmazonEC2ContainerServiceforEC2Role attached to allow ECS agent communication.
- User Data: Optionally, provide a script to automatically register the instance with your ECS cluster:
bash
```

- Configure Security Group:

```
Attach the security group you created earlier with rules for port 80 and any other required ports.
```

### _11.2 Connect the EC2 Instance to Your ECS Cluster:_

- Locate Your Key Pair File:

```
Ensure you have the private key file (your-key.pem) downloaded and accessible.
Store it in a location such as C:\Users\<YourUsername>\.ssh\your-key.pem.
```

- Set Permissions for the Key File:

> chmod 400 /path/to/your-key.pem

- SSH Command:

```
Use the public IP of the EC2 instance (available in the AWS EC2 dashboard under Instances > Instance Details): ssh -i /path/to/your-key.pem ec2-user@<public-ip>
```

- Check ECS Agent Status:

> sudo systemctl status ecs

- Register the Instance to Your ECS Cluster (if necessary):

> sudo echo "ECS_CLUSTER=your-cluster-name" >> /etc/ecs/ecs.config

> sudo systemctl restart ecs

- Verify the Instance in the ECS Console

```
Go to the AWS ECS Dashboard:
        Navigate to Clusters in the AWS ECS Console.
        Select your cluster.
```

- Check the "ECS Instances" Tab:

```
Ensure your EC2 instance is listed and in an active state.
```

### _12.2 Adjust Cluster configuration:_

- Update Packages ECS Agent

> sudo yum update -y

- Install Docker (if not already installed):

```
sudo yum install -y docker
sudo systemctl enable docker
sudo systemctl start docker
```

- Download the ECS Agent:

sudo docker pull amazon/amazon-ecs-agent:latest

> sudo docker pull amazon/amazon-ecs-agent:latest

- Run the ECS Agent:

```
sudo docker pull --platform linux/amd64 amazon/amazon-ecs-agent:latest
sudo docker run --platform linux/amd64 --name ecs-agent \
  --detach=true \
  --restart=on-failure:10 \
  --volume=/var/run/docker.sock:/var/run/docker.sock \
  --volume=/var/log/ecs/:/log \
  --volume=/var/lib/ecs/data:/data \
  --net=host \
  --env=ECS_CLUSTER=your-cluster-name \
  amazon/amazon-ecs-agent:latest
```

- Edit the ECS Configuration File:

> sudo mkdir -p /etc/ecs
> echo "ECS_CLUSTER=your-cluster-name" | sudo tee -a /etc/ecs/ecs.config

- Restart the ECS Agent:

> sudo docker restart ecs-agent

### _13.2 Create or Modify an IAM Role for ECS:_

- Navigate to the IAM Console:

- Create a New Role (or Use an Existing One):

```
In the left navigation pane, click Roles.
Click Create role to create a new role (or choose an existing role if you have one suitable for ECS).
```

- Select EC2 as the Trusted Entity:

```
In the "Select trusted entity" page, choose AWS service as the trusted entity type.
For the use case, select EC2 to allow EC2 instances to assume the role.
```

- Attach the AmazonEC2ContainerServiceforEC2Role Policy:

On the "Attach permissions policies" page, type AmazonEC2ContainerServiceforEC2Role in the search box.
Check the box next to AmazonEC2ContainerServiceforEC2Role to attach the policy.

- Name the Role:

> Give your role a meaningful name, such as ecs-instance-role or ecs-ec2-role.


- Click Next: Review.

- Click Create role to create the role.

- Attach the IAM Role to Your EC2 Instance.

```
Navigate to the EC2 Console, Select Your Instance, Modify the Instance IAM Role, In the IAM role dropdown, select the role you created earlier (e.g., ecs-instance-role), Click Update IAM role to attach the role to your EC2 instance.
```

### _14.2 Test the Deployment:_

- Find the Public IP Address of you EC2 :

> Check the ECS service details or EC2 instance public IP. In the form `http://<public-ip>`

- Adjust the public IP of your EC2 in the in the host field  of `test.ipynb` file:

> Run the test.ipynb file to get the prediction result's.