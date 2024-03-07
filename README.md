# Welcome to Our Modularized Physical Incorporated Neural Network Model
This model is developed for future building energy modeling.
So far, we have provided 4 basic functions:
1) Load prediction. You can use it to predict the Space load / HVAC Energy / Total Load of a building under different scenarios (Different climate/ city/ setpoint/ weather...)
2) Dynamic modeling. You can use it to model building environments such as humidity / temperature / IAQ(Ongoing). You can select different forecast horizon (15 min to 5 days) 
3) Energy Optimization. You can easily run Model Predictive Control based on the building dynamic model
4) Retrofit. You can estimate your building performance after R value / U value and SHGC retrofit only by operation data! 

I will provide a step-by-step tutorial as shown below:
We also have a public web app available: https://resitool.streamlit.app/ Which includes colab link so you can play with your dataset by a simple click. 

Section 1. Load prediction

Section 2. Dynamic modeling (Start from the most exciting part first! Rest coming soon, daily update!!)
For dynamic modeling, the model can be split into three major parts:
1) Dataset
   
   Outdoor air temperature, solar radiation, HVAC are required features
   Space air temperature is required target
   
3) Model

4) Train

Section 3. Energy Optimization

Section 4. Retrofit
