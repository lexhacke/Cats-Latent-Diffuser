import { useState } from 'react'
import { Routes, Route, Link, NavLink } from "react-router-dom"
import './App.css'

function NavBar() {
  return (
    <nav className="bg-amber-400 justify-between p-4">
      <ul className="flex gap-4 justify-center">
        <li><NavLink to="/" className="nav-link">Generate</NavLink></li>
        <li><NavLink to="/about" className="nav-link">About</NavLink></li>
        <li><a href="https://github.com/lexhacke">Github</a></li>
      </ul>
    </nav>
  )
}

function App() {
  return (
    <NavBar/>
  )
}

export default App
