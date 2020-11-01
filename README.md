# House-Prices-Advanced-Regression-Techniques
Sequential Forward Selection (SFS) model stacking

The goal of this project was to use EDA, data preprocessing, and linear and nonlinear models, as well as advanced model ensemble to predict house prices given 80 features of houses, and I will also try to interpret the linear models to find out which features are important to the house prices. The data was originally available on Kaggle.

To further improve generalizability and robustness over a single model, I employ model ensemble methods, i.e., to combine the predictions of several base estimators and make a more robust one. Compared with a simple model blending method (easy to implement), in which coefficients are empirically chosen for each base model, model stacking (or stacking regressor) enables such coefficients computed in a secondary level model. It thus allows a better performance.

However, to use all available base models in a stacked model may be not a good idea. (This will be demonstrated in the final evaluation section). On the other hand, it is computationally expensive to evaluate all possible combination of stacked models. Here, we use a heuristic sequential forward selection (SFS), aiming to find a (sub)optimal solution for the model stacking. See figure below.

