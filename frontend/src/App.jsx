import { useState } from 'react'
import Login from './components/Login'
import Results from './components/Results'
import Upload from './components/Upload'

export default function App() {
  const [loggedIn, setLoggedIn] = useState(!!localStorage.getItem('token'))
  const [result, setResult] = useState(null)

  function handleLogout() {
    localStorage.removeItem('token')
    setLoggedIn(false)
    setResult(null)
  }

  return (
    <div className="container">
      <h1>Government Document OCR Error Detection</h1>
      {!loggedIn ? (
        <Login onLoggedIn={() => setLoggedIn(true)} />
      ) : (
        <>
          <button type="button" onClick={handleLogout} style={{ background: '#999', float: 'right' }}>
            Log out
          </button>
          <Upload onResult={setResult} />
          <Results result={result} />
        </>
      )}
    </div>
  )
}
