import { useState } from 'react'
import { login, register } from '../api'

export default function Login({ onLoggedIn }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState('reviewer')
  const [mode, setMode] = useState('login') // 'login' | 'register'
  const [error, setError] = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    try {
      if (mode === 'register') {
        await register(email, password, role)
      }
      const { access_token: accessToken } = await login(email, password)
      localStorage.setItem('token', accessToken)
      onLoggedIn()
    } catch (err) {
      setError(err?.response?.data?.detail || 'Authentication failed.')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="card">
      <h2>{mode === 'login' ? 'Sign in' : 'Create account'}</h2>
      <input
        type="email"
        placeholder="Email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        required
      />
      {mode === 'register' && (
        <select value={role} onChange={(e) => setRole(e.target.value)}>
          <option value="reviewer">Reviewer (can approve/reject corrections)</option>
          <option value="admin">Admin (full access)</option>
          <option value="uploader">Uploader only (cannot approve/reject)</option>
        </select>
      )}
      <button type="submit">{mode === 'login' ? 'Login' : 'Register & login'}</button>
      <button
        type="button"
        onClick={() => setMode(mode === 'login' ? 'register' : 'login')}
        style={{ background: '#666' }}
      >
        {mode === 'login' ? 'Need an account? Register' : 'Have an account? Login'}
      </button>
      {error && <p className="error">{error}</p>}
    </form>
  )
}
