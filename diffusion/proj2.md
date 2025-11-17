# Proj 2

## Abstract
* What is the problem this paper addresses?
* Why is it an important problem?
* Why are current approaches insufficient?
* Methods: In this work, we develop an approach to address these deficiencies

This paper addresses the high computational cost of diffusion modelling. This is an important problem because pixel space diffusion requires forward passes over very high dimensional inputs (high resolution images), which makes training and inference very expensive, slows down image generation, and restricts maximum model size and input resolution. Furthermore, current diffusion models waste capacity modelling imperceptible, high frequency details in pixel space. To address these limitations, the authors develop Latent Diffusion Models, a class of 2-stage diffusion models that learn a mapping between pixel space and a low dimensional latent space, and perform diffusion only in that latent space. The resulting model is more computationally efficient and scalable, since all diffusion steps occur in latent space, and only a single forward pass of the decoder is required to reconstruct the final image in pixel space.

## Problem definition
* What question are you trying to solve?
* Observed and unobserved random variables?
* What is the goal of the project?

## Models and Methods
* Describe the model and inference algorithms.
* Graphical and generative model.
* What parameters do we estimate and how?
* What is the interpretation of those parameters, how do they solve the problem?

## Results and Validation
* What will your results show?
* How will you quanify how well your approach answered the question?
* What other models and methods will you compare against?
* How do you validate your answers and uncertainty?
* What figures/tables will you use?