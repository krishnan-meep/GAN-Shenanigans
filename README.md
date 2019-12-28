# GAN-Shenanigans

I'll be putting up all the GANs I've managed to get results from in this repository. They might be consolidated versions so you won't see a DCGAN separate from an SNGAN. Whenever I can they will be
notebooks for easy interactive use.

You need these before you run the notebooks
* torch, torchvision
* opencv

## List of my GANs
* **SNGAN with CIFAR**

	Spectral Normalization in the Discriminator, Self Modulation in the Generator, Hinge Loss.
	Results shown below are the model trained on just the DawgsOfCIFAR for 400 epochs.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/cifar_dogs_400.png)


* **SNGAN-ResNet Edition with CIFAR**

	Spectral Normalization and MiniBatch Discrimination in the Discriminator, Self Modulation in the Generator, Hinge Loss. Resblocks used in both networks.
	Results shown below are the model trained on just the BoidsOfCIFAR for 400 epochs.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/cifar_birds_400.png)


* More to come.....