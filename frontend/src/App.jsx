import { useState } from 'react'
import Login from './components/Login'
import Results from './components/Results'
import Upload from './components/Upload'
import { LogoutIcon, MoonIcon, ShieldCheckIcon, SunIcon } from './components/Icons'
import useTheme from './useTheme'

function ThemeToggle({ theme, onToggle }) {
  const goingDark = theme === 'light'
  const label = goingDark ? 'Switch to dark theme' : 'Switch to light theme'
  return (
    <button
      type="button"
      className="btn btn--ghost btn--icon"
      onClick={onToggle}
      aria-pressed={theme === 'dark'}
      title={label}
    >
      {goingDark ? <MoonIcon /> : <SunIcon />}
      <span className="sr-only">{label}</span>
    </button>
  )
}

export default function App() {
  const [loggedIn, setLoggedIn] = useState(!!localStorage.getItem('token'))
  const [result, setResult] = useState(null)
  const { theme, toggleTheme } = useTheme()

  function handleLogout() {
    localStorage.removeItem('token')
    setLoggedIn(false)
    setResult(null)
  }

  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>

      <header className="site-header">
        <div className="container site-header__inner">
          <div className="brand">
            <span className="brand__mark">
              <ShieldCheckIcon width={22} height={22} />
            </span>
            <span>
              <span className="brand__title">OCR Review Console</span>
              <span className="brand__subtitle">Government document digitization</span>
            </span>
          </div>

          <div className="header-actions">
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
            {loggedIn && (
              <button type="button" className="btn btn--sm" onClick={handleLogout}>
                <LogoutIcon width={16} height={16} />
                Log out
              </button>
            )}
          </div>
        </div>
      </header>

      <main
        id="main-content"
        tabIndex={-1}
        className={`container app-main${loggedIn ? '' : ' container--narrow app-main--centered'}`}
      >
        {!loggedIn ? (
          <Login onLoggedIn={() => setLoggedIn(true)} />
        ) : (
          <>
            <div>
              <h1>Document review</h1>
              <p className="muted">
                Upload a scanned document, then approve or reject each flagged field before the
                corrected text is stored.
              </p>
            </div>
            <Upload onResult={setResult} />
            <Results result={result} />
          </>
        )}
      </main>

      <footer className="site-footer">
        <div className="container">
          Context-driven OCR error detection · rules + dictionary + language-model signals · every
          decision written to the audit log.
        </div>
      </footer>
    </>
  )
}
