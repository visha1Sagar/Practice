## Description
This project demonstrates the use of gradients and Integrated Gradients as model interpretability techniques for a classification model. Both methods help to attribute a model's prediction to its input features, offering insight into how each feature influences the model's decision.

Gradients compute the gradient of the output with respect to the input, identifying which parts of the input are most important for a particular prediction.
Integrated Gradients improve upon standard gradients by addressing the issue of saturation. It does this by accumulating gradients along the path from a baseline (such as a black image) to the input image.
This project aims to compare both methods, providing visual explanations of how input features (e.g., pixels in an image) contribute to model predictions.

