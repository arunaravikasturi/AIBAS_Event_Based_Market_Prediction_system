Ownership  
This image is created by Shashank Kalaskar and Aruna Ravi.

Course Details  
Created for the course "M. Grum: Advanced AI-based Application Systems" by the Junior Chair for Business Information Science, esp. AI-based Application Systems at the University of Potsdam.

Image Purpose  
This Docker image contains the execution logic of the Event-Based Market Predictor.  
It loads the activation data, applies the trained ANN and OLS models from the knowledge base, and produces market predictions.

System Role  
The image is responsible only for runtime inference and does not contain any training data or trained models.  
It is designed to be used together with the learningBase, activationBase, and knowledgeBase images via docker-compose.

License  
This image and its contents are published under the AGPL-3.0 license.
