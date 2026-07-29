import { useEffect, useState } from 'react'
import { setUnauthorizedHandler } from './api'
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

  // An expired/invalid token surfaces as a 401 from any API call; route
  // that straight back to the login screen instead of a raw error message.
  useEffect(() => {
    setUnauthorizedHandler(handleLogout)
    return () => setUnauthorizedHandler(null)
  }, [])

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
            <div className="workspace-hero">
              <div className="workspace-hero__copy">
                <span className="workspace-hero__eyebrow">Secure review workspace</span>
                <h1>Turn scanned documents into trusted records.</h1>
                <p>
                  Upload a government document, inspect the context-aware findings, and approve
                  every correction before it becomes part of the final record.
                </p>
              </div>
              <div className="workspace-hero__steps" aria-label="Review workflow">
                <span className="workspace-step is-active">
                  <strong>01</strong>
                  Upload
                </span>
                <span className="workspace-step">
                  <strong>02</strong>
                  Review
                </span>
                <span className="workspace-step">
                  <strong>03</strong>
                  Export
                </span>
              </div>
            </div>
            <Upload onResult={setResult} />
            <Results result={result} onResult={setResult} />
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
