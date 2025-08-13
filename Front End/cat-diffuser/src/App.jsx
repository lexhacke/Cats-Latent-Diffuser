import { use, useState } from 'react'
import { Routes, Route, Link, NavLink } from "react-router-dom"
import './App.css'

function NavBar() {
  return (
    <nav className="bg-violet-950 justify-between p-4">
      <ul className="flex gap-16 justify-center">
        <li><NavLink to="/" className="clickable">Generate</NavLink></li>
        <li><NavLink to="/about" className="clickable">About</NavLink></li>
        <li><a href="https://github.com/lexhacke" className="clickable">Github</a></li>
      </ul>
    </nav>
  )
}

function Home() {
  async function generateCat({setUrl, setPhase}) {
    setPhase(1);
    const ENDPOINT_URL = "https://w2rosgyoyor4sofu.us-east-1.aws.endpoints.huggingface.cloud"
    const HF_TOKEN = ""
    const response = await fetch(ENDPOINT_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${HF_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({inputs: {}})
    })
    if (!response.ok) {
      setUrl("https://github.com/lexhacke/Cats-Latent-Diffuser/blob/main/Front%20End/cat-diffuser/error.jpg?raw=true")
      return 1;
    }
    const data = await response.json()
    const blob = new Blob([data['image']], { type: 'image/png' });
    const url = URL.createObjectURL(blob);
    setUrl(url);
    setPhase(2);
    return 0;
  }
  
  const [phase, setPhase] = useState(0); // 0 = init, 1 = await, 2 = recieved
  const [url, setUrl] = useState("");
  return <div>
    <NavBar />
    <div className='holder'>
      <h1 className="title">Cat Diffuser</h1>
      <div className='cat-box justify-center'>
        {phase !== 0 && <img src={url} className="cat-image"/>}
      </div>
      <button onClick={() => generateCat({setUrl, setPhase})} className="cool_button">
        {phase === 0 && "Generate!"}
        {phase >= 1 && "Generating..."}
      </button>
      </div>
    </div>
}

function About() {
  return <div>
    <NavBar />
    <h1>About</h1>
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
