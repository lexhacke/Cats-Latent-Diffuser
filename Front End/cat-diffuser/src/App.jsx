import { useState, useEffect } from 'react'
import { Routes, Route, Link, NavLink } from "react-router-dom"
import './App.css'

function NavBar() {
  return (
    <nav className="bg-violet-950 justify-between p-4">
      <ul className="flex gap-16 justify-center">
        <li><NavLink to="/" className="clickable">Generate</NavLink></li>
        <li><NavLink to="/about" className="clickable">About</NavLink></li>
        <li><a target="_blank" href="https://github.com/lexhacke/Cats-Latent-Diffuser" className="clickable">Github</a></li>
      </ul>
    </nav>
  )
}

function Home() {
  const [isLive, setLive] = useState(true);
  const [phase, setPhase] = useState(0); // 0 = init, 1 = await, 2 = recieved
  const [url, setUrl] = useState("");
  const [startTime, setTime] = useState(0);
  const [delta, setDelta] = useState(0);

  async function generateCat() {
    if (phase == 1) return;
    setPhase(1);
    try {
      setTime(performance.now());
      console.log("Request Sent");
      const response = await fetch("https://uibdebrl6rqpfv2rly3ql5mqoq0wwqmq.lambda-url.us-east-2.on.aws/", {
        method: "POST",
        body: JSON.stringify({"key": "loveandlighttv"})
      });
      if (!response.ok) {
        throw new Error('Goddamn it');
      }
      const rawBytes = await response.json();
      const bytes = atob(rawBytes);
      const byteArr = Array.from(bytes, c => c.charCodeAt(0));
      const matrix = new Uint8Array(byteArr);
      const blob = new Blob([matrix], { type: 'image/png' });
      const url = URL.createObjectURL(blob);
      setUrl(url);
      setPhase(2);
    } catch (error) {
      alert(error)
      setUrl("https://github.com/lexhacke/Cats-Latent-Diffuser/blob/main/Front%20End/cat-diffuser/error.jpg?raw=true");
      setPhase(2);
      setLive(false);
    }
  }

  useEffect(() => {
    let intervalId;
    if (phase === 1) {
      intervalId = setInterval(() => {
        setDelta(performance.now() - startTime);
      }, 10);
    }
    return () => clearInterval(intervalId);
  }, [phase, startTime]);

  return <div>
    <NavBar />
    <div className='holder'>
      <h1 className="title">Cat Diffuser</h1>
      <div className='cat-box justify-center'>
        {phase === 1 && <div className="flex flex-col justify-center items-center">
                        <p className='text-sm'>{Math.floor(delta/1000)}s</p>
                        <div className="animate-spin rounded-full h-15 w-15 border-t-2 border-b-2 border-violet-500"></div>
                        {(delta > 250*1000) && <p>This usually takes ~5 min sorry :(</p>}
                      </div>}
        {phase === 2 && <img src={url} className="cat-image"/>}
      </div>
      <br/>
      { isLive ? 
      <button onClick={generateCat} className={phase != 1 ? "cool_button": "cool_button_active"}>
        { phase != 1 ? "Generate!" : "Generating..." }
      </button>
        : 
        <p>
        It seems like the endpoint is inactive. It's set to sleep when it doesn't get accessed for a while, so give it a minute or so and it should be working.
        If its been 5 minutes and you're <em>still</em> seeing this message, I probably ran out of money on my AWS (or huggingface) account and the endpoint got deleted.
        <br/><br/>
        Sorry for any inconvenience! If you want to run it yourself, you can find the code on 
        <a target="_blank" className='minilink' href="https://huggingface.co/detectivejoewest/cat_diffusion"> Huggingface</a>. Best of luck! 
        <br/>
        <br/>- Lex Hackett
        </p>
      }
      </div>
    </div>
}

function About() {
  return <div>
    <NavBar />
    <h1 className="title">About Cat Diffuser</h1>
    <p className='text-center max-w-3xl mx-auto'>
      Cat Diffuser is my custom deployment of a Latent DDIM diffusion model trained exclusively on the 
      <a className="minilink" target="_blank" href='https://www.kaggle.com/datasets/crawford/cat-dataset'> Crawford Cats</a> dataset.
      The model takes generates a completely new cat picture from scratch every time you click generate.
      The model architecture is a pretty vanilla U-Net, similar to early 1.X Stable Diffusion models, albeit much smaller. It consists of a ResNet-style downsampling path, a bottleneck with spatial attention, and a skip-connected upsampling path
      where the image is progressively refined through a series of Residual Blocks (where equiresolution blocks are connected via skip connections).
      While the images you see are 512x512, the model was actually trained on 64x64x4 latent images using the 
      <a className='minilink' target="_blank" href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"> Stable Diffusion 1.5 Variation Autoencoder</a>.
      Why did I use latent diffusion, you ask? Well, firstly, it’s incredibly less resource-intensive to train a model on 64×64×4 inputs 
      than on 512×512×3 inputs. Period. That’s about a 4,800% reduction in raw input size.  Furthermore, autoencoders—especially those trained with 
      adversarial losses, think PatchGAN—tend to “hallucinate” fine textures and features that might have been lost if we trained a diffusion model 
      directly in pixel space, allowing us to actually recover smaller details and sharpness 
      (<a target="_blank" className='minilink' href="https://arxiv.org/pdf/2112.10752">Romach et al.</a>).
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
      <a className='minilink' target="_blank" href="https://www.github.com/lexhacke"> Github </a> | 
      <a className='minilink' target="_blank" href="https://lex.hackett@hotmail.com"> Hotmail </a>
    </p>
    <br/>
    <hr className="my-4 h-0.5 border-0 bg-white"/>
    <br/>
    </div>
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/about" element={<About />} />
      <Route path="*" element={<h1>404 Not Found</h1>} />
    </Routes>
  )
}

export default App
