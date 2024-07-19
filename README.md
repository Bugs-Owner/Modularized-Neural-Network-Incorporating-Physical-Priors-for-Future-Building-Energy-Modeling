# Modularized Neural Network Incorporating Physical Priors for Future Building Energy Modeling

![fx1_lrg](https://github.com/user-attachments/assets/c82ea76a-494a-4c2b-a593-0cc729a97d36)

This repository proposed a modularized neural network incorporating physical priors (ModNN) for Future Building Energy Modeling, which can be used for 

1) load prediction,
![image](https://github.com/user-attachments/assets/49d03cfb-6519-42e5-b702-265e393d8a33)

2) dynamic modeling,
![ModNN](https://github.com/user-attachments/assets/bc144fbd-bff0-4988-ae2b-5f7c1a991480)

3) retrofit
![gr4_lrg](https://github.com/user-attachments/assets/2aee36ec-1bf0-46c4-a732-7e973ab213f0)

4) energy optimization:
![image](https://github.com/user-attachments/assets/e2b3925d-1964-41bf-9ae8-7bd1c9d7def8)
 
The incorporation of physical knowledge can be summarized in four key points: 

1) Heat Balance-Inspired Modularization
   
We incorporated physical knowledge by modularizing the model structures to create a heat balance framework. Specifically, we developed distinct neural network modules to estimate each unique heat transfer term of the dynamic building system.
![gr1_lrg](https://github.com/user-attachments/assets/6b941d58-bcee-4de7-ab24-af456ae48ce4)

2) State-Space-Inspired Encoder-Decoder Structure
   
An encoder is designed to extract historical information, a current cell measures data from the current time step, and a decoder predicts system responses based on future system inputs and disturbances.

3) Physically Consistent Model Constraints
   
We introduce physical consistency constraints to ensure the model responds appropriately to given inputs. For example, the conduction heat flux through a wall decreases as the R-value increases, and indoor air temperature decreases with an increasing HVAC cooling load.

4) Lego Brick-Inspired Modular Design
   
We connect different modules based on physical typology, allowing for multiple-building applications through model sharing and inheritance.

## Instruction: 
Step 1: Change your dataset to fit the column name

Step 2: Run run.py file

Done! Yes, you only need two steps. More detailed information coming soon~

## Publication: 
Zixin Jiang, Bing Dong,
Modularized neural network incorporating physical priors for future building energy modeling,
Patterns,2024,101029, ISSN 2666-3899,
https://doi.org/10.1016/j.patter.2024.101029.

## Contact
Please send me an email: zjiang19@syr.edu if you have any questions. Thank you~
