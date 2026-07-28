import { useRef, useState } from 'react'
import { login, register } from '../api'
import { AlertIcon, ShieldCheckIcon } from './Icons'

const MODES = [
  { id: 'login', label: 'Sign in' },
  { id: 'register', label: 'Create account' },
]

export default function Login({ onLoggedIn }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState('reviewer')
  const [mode, setMode] = useState('login') // 'login' | 'register'
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const tabRefs = useRef({})

  const isRegister = mode === 'register'

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      if (isRegister) {
        await register(email, password, role)
      }
      const { access_token: accessToken } = await login(email, password)
      localStorage.setItem('token', accessToken)
      onLoggedIn()
    } catch (err) {
      setError(err?.response?.data?.detail || 'Authentication failed. Check your credentials.')
    } finally {
      setBusy(false)
    }
  }

  // Roving focus between the two tabs, per the WAI-ARIA tabs pattern.
  function handleTabKeyDown(e) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    e.preventDefault()
    const next = mode === 'login' ? 'register' : 'login'
    setMode(next)
    setError('')
    tabRefs.current[next]?.focus()
  }

  return (
    <section className="card" aria-labelledby="auth-heading">
      <div className="stack">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span className="brand__mark">
            <ShieldCheckIcon width={22} height={22} />
          </span>
          <div>
            <h1 id="auth-heading" style={{ fontSize: '1.35rem' }}>
              {isRegister ? 'Create an account' : 'Sign in'}
            </h1>
            <p className="text-sm muted">Authorized reviewers only.</p>
          </div>
        </div>

        <div className="segmented" role="tablist" aria-label="Authentication mode">
          {MODES.map((m) => (
            <button
              key={m.id}
              ref={(el) => {
                tabRefs.current[m.id] = el
              }}
              type="button"
              role="tab"
              id={`auth-tab-${m.id}`}
              aria-selected={mode === m.id}
              aria-controls="auth-panel"
              tabIndex={mode === m.id ? 0 : -1}
              className="segmented__option"
              onClick={() => {
                setMode(m.id)
                setError('')
              }}
              onKeyDown={handleTabKeyDown}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* role="tabpanel" belongs on a generic container — it isn't a valid
            role for <form>, which carries its own implicit semantics. */}
        <div id="auth-panel" role="tabpanel" aria-labelledby={`auth-tab-${mode}`}>
          <form className="stack" onSubmit={handleSubmit}>
            <div className="field">
              <label className="field__label" htmlFor="auth-email">
                Email address
              </label>
              <input
                id="auth-email"
                className="input"
                type="email"
                inputMode="email"
                autoComplete="email"
                placeholder="reviewer@example.gov"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                aria-invalid={error ? 'true' : undefined}
                aria-describedby={error ? 'auth-error' : undefined}
                required
              />
            </div>

            <div className="field">
              <label className="field__label" htmlFor="auth-password">
                Password
              </label>
              <input
                id="auth-password"
                className="input"
                type="password"
                autoComplete={isRegister ? 'new-password' : 'current-password'}
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                aria-invalid={error ? 'true' : undefined}
                aria-describedby={error ? 'auth-error' : undefined}
                required
              />
            </div>

            {isRegister && (
              <div className="field">
                <label className="field__label" htmlFor="auth-role">
                  Account role
                </label>
                <select
                  id="auth-role"
                  className="select"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  aria-describedby="auth-role-hint"
                >
                  <option value="reviewer">Reviewer</option>
                  <option value="admin">Admin</option>
                  <option value="uploader">Uploader</option>
                </select>
                <p id="auth-role-hint" className="field__hint">
                  Reviewers and admins can approve or reject corrections. Uploaders can only upload
                  documents and view results.
                </p>
              </div>
            )}

            <button type="submit" className="btn btn--primary btn--block" disabled={busy}>
              {busy && <span className="spinner" aria-hidden="true" />}
              {busy ? 'Signing in…' : isRegister ? 'Create account & sign in' : 'Sign in'}
            </button>

            <div aria-live="assertive">
              {error && (
                <p className="alert alert--error" id="auth-error" role="alert">
                  <AlertIcon className="alert__icon" width={16} height={16} />
                  <span>{error}</span>
                </p>
              )}
            </div>
          </form>
        </div>
      </div>
    </section>
  )
}
