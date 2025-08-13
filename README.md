<h1>Cat Latent Diffusion</h1>

Cat Diffuser is my custom deployment of a Latent DDIM diffusion model trained exclusively on the Crawford Cats dataset. 
The model takes generates a completely new cat picture from scratch every time you click generate. The model architecture 
is a pretty vanilla U-Net, similar to early 1.X Stable Diffusion models, albeit much smaller. It consists of a ResNet-style
downsampling path, a bottleneck with spatial attention, and a skip-connected upsampling path where the image is progressively
refined through a series of Residual Blocks (where equiresolution blocks are connected via skip connections). While the images
you see are 512x512, the model was actually trained on 64x64x4 latent images using the Stable Diffusion 1.5 Variational 
Autoencoder. Why did I use latent diffusion, you ask? Well, firstly, it’s incredibly less resource-intensive to train a 
model on 64×64×4 inputs than on 512×512×3 inputs. Period. That’s about a 4,800% reduction in raw input size. Furthermore, 
autoencoders—especially those trained with adversarial losses, think PatchGAN—tend to “hallucinate” fine textures and features 
that might have been lost if we trained a diffusion model directly in pixel space, allowing us to actually recover smaller 
details and sharpness (Romach et al.).
