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

The Latent Diffusion Model is made up of 2 parts: an autoencoder for perceptual compression and a denoising U-Net for diffusion.

### Autoencoder
The authors tested 2 autoencoders, a VAE (Variational Auto Encoder) and VQ-GAN (Vector-Quantized Generative Adversarial Network). Keep in mind that the purpose of the autoencoder is to compress images into a smaller latent space so diffusion can run more efficiently, while losing as little image fidelity as possible. Both autoencoders are trained first (via SGD), separately from diffusion model and fixed during diffusion model training.

#### VAE
VAEs have 2 parts: an encoder that probabilistically compresses inputs into latent factors (or codes), and a decoder that reconstructs the sampled latent codes back into image space. Classic VAEs minimize the following objective: $$L = -\mathbb{E}_{z \sim q_\phi(z | x)} \log p_\theta(x | z) + \text{KL} \left[ q_\phi(z | x) || p(z) \right]$$

The first term $L_{recon} = -\mathbb{E}_{z \sim q_\phi(z | x)} \log p_\theta(x | z)$ is the reconstruction loss (how likely is the image $x$ that was reconstructed from the sampled latent code $z$?), which encourages the VAE to output realistic images.

The second term $L_{reg} = \text{KL} \left[ q_\phi(z | x) || p(z) \right]$ penalizes the VAE for latents $z$ that deviate from the prior $p(z)$. A commonly used prior is $p(z) = N(0, I)$, which encourages the model to learn a disentangled latent space that essentially decomposes the high dimensional input space into the product of independent Gaussian features.

$\text{VAE generative model:}$ <br>
$ \cdot \text{ sample latents: } z \sim N(0, I)$ <br>
$ \cdot \text{ reconstruct (decode) latents into images: } x \sim p_\theta(x | z)$ <br>

![vae graphical model](images/vae_graphical_model.png)

The authors use a modified VAE with $$L_{recon} = \lambda_{L1} ||x - \mu_\theta(z)||_1 + \lambda_{per} \text{LPIPS}(x, \mu_\theta(z)) + \lambda_{adv} L_{adv}$$

The first term $\lambda_{L1} ||x - \mu_\theta(z)||_1$ is just the L1 loss between the image $x$ and the reconstructed image $\mu_\theta(z)$. The authors chose to use an L1 instead of L2 loss because they found that using an L1 loss made images less blurry (L2 is minimized by the mean value, which for images, is gray and blurry, while L1 is minimized by the median value).

The second term $\lambda_{per} \text{LPIPS}(x, \mu_\theta(z))$ is the perceptual loss term. LPIPS is the Learned Perceptual Image Patch Similarity defined as $\text{LPIPS}(x, \hat x) = \sum_l w_l ||\phi_l(x) - \phi_l(\hat x)||^2$, where $\phi_l(x)$ are the intermediate layer activations of some pretrained neural network (VGG-16). This loss term encourages the reconstruction to have similar deeper features to the original image, features at a higher level of abstraction compared to pixels (like textures) that the pretrained neural network learned.

The final term $L_{adv} = - D(\mu_\theta(z))$ is the patch-based adversarial loss, where $D$ is the GAN discriminator that must predict if the patch is real or generated. This patch-based adversarial loss forces the VAE decoder to generate sharp, realistic looking patches, patches indistinguishable from ones coming from real images.

#### VQ-GAN
The VQ-GAN is very similar to the VAE modified with adversarial loss as described above. The major difference is that the encoded latents are no longer continuous, but instead come from a discrete set of learnable embeddings (codewords). Encoded image latents are quantized to the closest (based on L2 distance) codewords, and the decoder only ever uses the quantized codewords to reconstruct the input image. The only additional loss term is $L_{quant} = || \text{sg}(z_q) - z ||^2$, where $z_q$ is the quantized latent and $\text{sg}$ is the stop gradient operator (codewords are updated using EMA and not gradients in more recent versions of VQ-VAE). This loss term is called the commitment loss, and pushes the encoder to use latents in the codebook (commit to them).

$\text{VQ-GAN generative model:}$ <br>
$ \cdot \text{ sample latent codewords: } z \in \{z_1 .. z_D\}$ <br>
$ \cdot \text{ reconstruct (decode) latent codewords into images: } x \sim p_\theta(x | z)$ <br>

![vq gan graphical model](images/vq_gan_graphical_model.png)

### Diffusion model
The backbone of the diffusion model is a residual U-Net. Time embeddings are added to intermediate activations to condition the diffusion model on time. Domain-specific conditioning is done through cross-attention at intermediate U-Net layers. Domain specific encoders (ie unmasked transformer for text) embed token-based conditioning information to be fed into cross-attention.

## Results and Validation
* What will your results show?
* How will you quantify how well your approach answered the question?
* What other models and methods will you compare against?
* How do you validate your answers and uncertainty?
* What figures/tables will you use?