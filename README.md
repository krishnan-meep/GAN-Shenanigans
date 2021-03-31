# GAN-Shenanigans

I'll be putting up all the GANs I've managed to get results from in this repository. They might be consolidated versions so you won't see a DCGAN separate from an SNGAN. Whenever I can they will be
notebooks for easy interactive use.

You need these before you run the notebooks
* torch, torchvision
* opencv

## List of my GANs
* **SNGAN with CIFAR**

	Spectral Normalization in the Discriminator, Self Modulation in the Generator, Hinge Loss.
	Results shown below are of the model trained on just the DawgsOfCIFAR for 400 epochs.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/cifar_dogs_400.png)


* **SNGAN-ResNet Edition with CIFAR**

	Spectral Normalization and MiniBatch Discrimination in the Discriminator, Self Modulation in the Generator, Hinge Loss. Resblocks used in both networks.
	Results shown below are of the model trained on just the BoidsOfCIFAR for 400 epochs.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/cifar_birds_400.png)

* **SNGAN-DenseNet Edition with CIFAR**

	Spectral Normalization and MiniBatch Discrimination in the Discriminator, Self Modulation in the Generator, Hinge Loss. Dense blocks used in both networks. Normal down/upblocks used as transition
	layers with a batch norm and activation before them. Same as the above two notebooks in all other
	regards. 
	Results shown below are of the model trained on Deer folk for 300 epochs.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/cifar_deer_300.png)

* **CycleGAN - Ukiyo-e Paintings & Fruits**

	UNet-Generators with Resblocks in the middle, basic discriminators (PatchGAN is available but it doesn't seem to fare as well as a vanilla discriminator).

	Results shown below took 100 epochs. It's slow going training four networks at once.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/cycgan_uki_fruits.png)

* **pix2pix - Colorizing Fruit**

	Doesn't work yet, not sure why, uses a UNet-Generator and a PatchGAN discriminator. 

* **WGAN-GP + SNGAN-ResNet for Pokemon**

	An alternate version of the SNGAN-ResNet model above with Wasserstein Loss and Gradient Penalty used instead of hinge loss. As of the moment, the architectures haven't been changed, meaning that it's using the CIFAR architecture, which is pretty small. SpecNorm and Self Modulation are present. Seems to produce the best results and it converges beautifully.

	Here are Pokemon generated with a 1000 epochs (64x64 images). Could still improve on the geometrical features but the color palettes and general shapes are swell.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/wgangp_poke_1000.png)

* **Progressive GAN**

	Progressively growing a basic DCGAN, it takes a while to run so not sure if it works yet.

* **Conditional SNGAN with Word Embeddings**

	Conditional GAN using continous labels instead, with GLoVe word embeddings in this particular case.

* **StarGAN**

	A more generalized version of the CycleGAN using just a single generator with conditioning.

* **SNGAN-FractalNet Edition with CIFAR**

	Spectral Normalization in the Discriminator, Self Modulation in the Generator, Wasserstein Loss with Gradient Penalty. Fractal blocks used in both networks. Look at [this](https://arxiv.org/abs/1605.07648) paper to know more about Fractal blocks.
	The horses shown below were obtained after a 100 epochs.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/cifar_horses_100.png)

* **GauGAN with Binary Thresholded Images**

	Patch Discriminator, SPADE Generator with Binary Thresholded images for modulation instead of segmentation maps, Hinge Loss, SpecNorm in both networks. Trained it on a set of landscape images.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/gaugan_bin.png)

* **RealnessGAN**

	Considers a distribution instead of a single scalar output from the discriminator. Real and fake anchor (discrete) distributions with KL Divergence Loss. Very basic Generator and Discriminator
	with SpectralNorm.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/cifar_realness.png)

* **Evolutionary GAN**

	Mutates a generator parent using different loss function and picks the best performing child for the next iteration.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/evo_cifar.png)

* **A Really Mixed Bag**

	Uses a Vision Transformer for the discriminator, dividing 32x32 CIFAR images into 8x8 patches. Uses a StyleGAN like generator. Uses Relativistic Averaging Least Squares Loss. What a mouthful.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/transformer_disc_results.png)


* More to come.....