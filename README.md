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

* **SNGAN-DenseNet Edition with CIFAR**

	Spectral Normalization and MiniBatch Discrimination in the Discriminator, Self Modulation in the Generator, Hinge Loss. Dense blocks used in both networks. Normal down/upblocks used as transition
	layers with a batch norm and activation before them. Same as the above two notebooks in all other
	regards. 
	Results shown below are the model trained on Deer folk for 300 epochs.

	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/cifar_deer_300.png)

* CycleGAN - Ukiyo-e Paintings & Fruits

	UNet-Generators with Resblocks in the middle, PatchGAN discriminators.
	Results shown below took 100 epochs. It's slow going training four networks at once.


	![](https://github.com/krishnan-meep/GAN-Shenanigans/blob/master/images_results/cycgan_uki_fruits.png)

* More to come.....