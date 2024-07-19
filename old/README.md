# Modularized Neural Network Incorporating Physical Priors for Future Building Energy Modeling

This repository proposed a modularized neural network incorporating physical priors (ModNN) for Future Building Energy Modeling, which can be used for 1) load prediction, 2) dynamic modeling, 3) retrofit, and 4) energy optimization. 
The incorporation of physical knowledge can be summarized in four key points: 

1) Physics-inspired modularization (inspired by heat balance equations where we use different NNs to represent each heat balance term),
2) Physics-inspired model structure (inspired by the state space model, an encoder is designed to extract historical information, a current cell is designed to measure data from the current time step, and a decoder is designed to predict the system responses based on future inputs)
3) Physics-inspired model constraints (we introduce physical consistency constraints to ensure that the model responds appropriately to given model inputs)
4) Physics-inspired model assembly (inspired by Lego brick, you can combine different module for multiple-building applications through model sharing and inheritance)


## Instruction

Feel free to try on your dataset, you need to change your column names and then you can start to play with Run.py file
More detailed information is coming soon, I will try to update at least once a week.

## Related publications


## Contact
Please send me an email: zjiang19@syr.edu if you have any questions. Thank you~
