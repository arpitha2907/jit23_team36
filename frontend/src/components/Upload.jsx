import { useState } from 'react'
import { getResult, getStatus, uploadDocument } from '../api'

// First-ever upload after the backend starts has to download/load the BERT
// model (can take a minute or two); every upload after that is fast. Poll
// generously so a slow first run doesn't look like a failure.
const POLL_INTERVAL_MS = 2000
const POLL_ATTEMPTS = 90 // ~3 minutes

export default function Upload({ onResult }) {
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [pendingId, setPendingId] = useState(null)

  async function pollUntilReady(id) {
    setPendingId(id)
    for (let i = 0; i < POLL_ATTEMPTS; i++) {
      const { status } = await getStatus(id)
      if (status === 'needs_review' || status === 'approved') {
        const result = await getResult(id)
        onResult(result)
        setMessage('')
        setPendingId(null)
        return
      }
      if (status === 'failed') {
        setMessage('Processing failed on the server. Check `docker compose logs backend`.')
        setPendingId(null)
        return
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS))
    }
    setMessage(
      'Still processing on the server (this can happen on the very first ' +
        'upload while the model loads). Click "Check again" below once more ' +
        'time has passed — no need to re-upload.'
    )
  }

  async function handleUpload(e) {
    e.preventDefault()
    if (!file) return
    setBusy(true)
    setMessage('Uploading and processing...')
    try {
      const doc = await uploadDocument(file)
      await pollUntilReady(doc.id)
    } catch (err) {
      setMessage('Upload failed: ' + (err?.response?.data?.detail || err.message))
    } finally {
      setBusy(false)
    }
  }

  async function handleCheckAgain() {
    if (!pendingId) return
    setBusy(true)
    setMessage('Checking...')
    try {
      await pollUntilReady(pendingId)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={handleUpload} className="card">
      <h2>Upload a document image</h2>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
        required
      />
      <button type="submit" disabled={busy}>
        {busy ? 'Processing...' : 'Upload'}
      </button>
      {pendingId && !busy && (
        <button type="button" onClick={handleCheckAgain}>
          Check again
        </button>
      )}
      {message && <p>{message}</p>}
    </form>
  )
}
