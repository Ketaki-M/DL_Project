# DL_Project

Accurate angle prediction is a vital component in modern technologies such as autonomous vehicles, robotics, and navigation systems. In self-driving cars, for example, steering angle estimation ensures smooth lane changes, precise turns, and safe navigation in complex environments. Industry leaders like Waymo and Tesla have showcased how AI-driven angle prediction can improve vehicle control and passenger safety, while platforms like the Udacity Self-Driving Car Simulator have made it possible for developers and researchers to experiment with these models in a simulated environment. This project — Angle Prediction System — leverages deep learning to predict angles from sensor or image-derived data, offering a scalable and reliable approach for real-time decision-making. 

By integrating data preprocessing, neural network modeling, and performance evaluation, the system demonstrates how AI can be applied to critical control tasks where precision is essential. Beyond autonomous cars, this solution is adaptable for use in drones, robotic arms, industrial automation, and smart devices requiring accurate orientation awareness.
1)Features

This project is a reimplementation of NVIDIA’s end-to-end deep learning approach for steering angle prediction as described in their research paper "End to End Learning for Self-Driving Cars". NVIDIA’s work demonstrated that a convolutional neural network (CNN) could directly map raw input data (such as camera images) to steering commands, eliminating the need for manually designed perception pipelines. Accurate angle prediction is a key component in autonomous driving, enabling smooth lane changes, precise turns, and safe navigation. 

Industry leaders like Waymo and Tesla have proven the effectiveness of such AI-driven control systems, while the Udacity Self-Driving Car Simulator has become a popular platform for testing and training steering angle models in a controlled environment.

The Angle Prediction System presented here follows NVIDIA’s methodology, adapted for a custom dataset and workflow. The project includes data preprocessing, model training using deep learning regression, and evaluation using metrics like MAE and RMSE. This reimplementation demonstrates how AI can be applied to real-time decision-making tasks in self-driving cars, and it can be extended to other applications such as drones, robotic arms, and orientation-aware IoT devices.

1)KEY POINTS

* Preprocessing and normalization for stable model training
* Neural network architecture optimized for regression tasks
* Evaluation using MAE and RMSE metrics
* Visualization of predicted vs. actual values
* Scalable for real-time prediction use cases

2)Tech Stack

* **Programming Language:** Python
* **Libraries:** TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn
* **Development Environment:** Jupyter Notebook

3)Workflow

1. Load and preprocess dataset
2. Train deep learning regression model
3. Evaluate using error metrics
4. Visualize results

4)Applications

* Robotic arm position control
* Orientation estimation in IoT devices
* Camera and sensor calibration
  
<img height="400" alt="Screenshot 2025-04-16 at 12:07:29 PM" src="https://github.com/user-attachments/assets/02e852fc-0e70-4301-9595-1caf6aaaeb78" />

<img height="500" alt="Screenshot 2025-04-16 at 12:10:51 PM" src="https://github.com/user-attachments/assets/5d12e53a-e008-45ed-aa0d-8fb37ee592bd" />





  


