<h1>Cat Latent Diffusion</h1>

 Cat Diffuser is my custom deployment of a Latent DDIM diffusion model trained exclusively on the 
      <a className="minilink" target="_blank" href='https://www.kaggle.com/datasets/crawford/cat-dataset'> Crawford Cats</a> dataset.
      The model takes generates a completely new cat picture from scratch every time you click generate.
      The model architecture is a pretty vanilla U-Net, similar to early 1.X Stable Diffusion models, albeit much smaller.
      The exact model architecture is as follows:
      <ol>
      <li>Firstly, I encoded the timestep using the standard sinusoidal embedding from Vaswani et al. and expanded the resulting 1×512 vector into a 512×64×64 tensor by tiling across spatial dimensions. This tensor was then concatenated to the latent representation, so each latent pixel had a direct encoding of the timestep information.</li>
      <li>I then passed the latent image through a conv2d layer that reduced the 516x64x64 dimensional latent to a 32x64x64 latent</li>
      <li>I then passed the image through a U-net ResNet-style downsampling path (each layer consisting of 2 residual blocks and a strided convolution), the Conv2d filter counts were as follows: 32, 64, 256, 512. All intermediate resolutions were saved in a skip connection stack</li>
      <li>The bottleneck consisted of a simple ResNet-style Residual Block, followed by Spatial Self-Attention, and another Residual Block</li>
      <li>Lastly, the U-net upsampling layer followed the same Conv2d filter counts, albeit reversed (each Layer consisting of a skip concatenation, followed by bilinear upsampling and 2 Residual Blocks).</li>
      <li>Finally, a Conv2d layer projects the 32x64x64 image back to the autoencoder's 4x64x64 latent space using a 4-filter convolution</li>
      </ol>
      While the images you see are 512x512, the model was actually trained on 64x64x4 latent images using the 
      <a className='minilink' target="_blank" href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"> Stable Diffusion 1.5 Variational Autoencoder</a>.
    </p>
    <br/>
    <h1 className='subtitle'>What did I change?</h1>
    <p className='text-center max-w-3xl mx-auto'>
      You might notice that inference is really slow. Being a college student at the moment,
      I could only afford to train the model on a T4 for 22 hours, costing me a little under 20 bucks.
      You might notice some images don't really look exactly like cats, but you'll get the occassional
      really pretty one, like these:
    </p>
    <img className='kitties' src="https://github.com/lexhacke/Cats-Latent-Diffuser/blob/main/Model/cutest_cats.png?raw=true"/>
    <br/>
    <p className='text-center max-w-3xl mx-auto'>
      Knowing that I don't have the resources to make the perfect model, I decided to use a little neat trick I figured out after tinkering with CLIP embeddings for my
      Diffusion Transformer project I'm working on right now. During inference, I generate ~10 images with the model, which doesn't turn out to be too expensive my huggingface endpoint
      natively supports batching (i.e. parallel processing). I then use CLIP to score each image by embedding the image and the text "a photo of a cat" and taking the image with
      the highest cosine similarity! I thought I was so smart for figuring this out, but it turns out its been out for a while, called "CLIP reranking / best-of-N sampling". Bummer.
      But at least it makes my images a little better!
    </p>
    <br/>
    <p className='text-center max-w-3xl mx-auto'>
      To be honest, I don't really know how to use React, so this is my first time using it. I really see myself as more of a researcher or machine learning engineer, but seeing 
      <em> Javascript, React, Full Stack</em> as required skills for all these internship positions I'm applying to really forced me to bite the bullet. I hope I didn't accidentally expose
      any of my API keys or anything like that, but if I did, please let me know! I really don't want to get hacked.
      <br/><br/>
      By the way, can find me here:
      <br/>
      <a className='minilink' target="_blank" href="https://www.linkedin.com/in/lex-hackett-2b89612b3/"> LinkedIn </a> |
      <a className='minilink' target="_blank" href="https://lex.hackett@hotmail.com"> Hotmail </a>
