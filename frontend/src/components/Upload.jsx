import { useEffect, useRef, useState } from 'react'
import { getResult, getStatus, uploadDocument } from '../api'
import { AlertIcon, CheckCircleIcon, CloseIcon, ImageIcon, InfoIcon, UploadIcon } from './Icons'

// First-ever upload after the backend starts has to download/load the BERT
// model (can take a minute or two); every upload after that is fast. Poll
// generously so a slow first run doesn't look like a failure.
const POLL_INTERVAL_MS = 2000
const POLL_ATTEMPTS = 90 // ~3 minutes

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function Upload({ onResult }) {
  const [file, setFile] = useState(null)
  const [docType, setDocType] = useState('generic')
  const [ocrLanguages, setOcrLanguages] = useState('eng')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState(null) // { tone, text }
  const [pendingId, setPendingId] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [thumbUrl, setThumbUrl] = useState(null)
  const inputRef = useRef(null)

  // Local thumbnail for the chosen file; revoked whenever it's replaced.
  useEffect(() => {
    if (!file) {
      setThumbUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setThumbUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  async function pollUntilReady(id) {
    setPendingId(id)
    for (let i = 0; i < POLL_ATTEMPTS; i++) {
      const { status } = await getStatus(id)
      if (status === 'needs_review' || status === 'approved') {
        const result = await getResult(id)
        onResult(result)
        setMessage({ tone: 'success', text: 'Processing complete — review the fields below.' })
        setPendingId(null)
        return
      }
      if (status === 'failed') {
        setMessage({
          tone: 'error',
          text: 'Processing failed on the server. Check `docker compose logs backend`.',
        })
        setPendingId(null)
        return
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS))
    }
    setMessage({
      tone: 'warning',
      text:
        'Still processing on the server (this can happen on the very first upload while the ' +
        'model loads). Select “Check again” below once more time has passed — no need to re-upload.',
    })
  }

  async function handleUpload(e) {
    e.preventDefault()
    if (!file || busy) return
    setBusy(true)
    setMessage({ tone: 'info', text: 'Uploading and running the OCR pipeline…' })
    try {
      const doc = await uploadDocument(file, docType, ocrLanguages)
      await pollUntilReady(doc.id)
    } catch (err) {
      setMessage({
        tone: 'error',
        text: 'Upload failed: ' + (err?.response?.data?.detail || err.message),
      })
    } finally {
      setBusy(false)
    }
  }

  async function handleCheckAgain() {
    if (!pendingId) return
    setBusy(true)
    setMessage({ tone: 'info', text: 'Checking processing status…' })
    try {
      await pollUntilReady(pendingId)
    } catch (err) {
      setMessage({
        tone: 'error',
        text: 'Check failed: ' + (err?.response?.data?.detail || err.message),
      })
    } finally {
      setBusy(false)
    }
  }

  function selectFile(candidate) {
    if (!candidate) return
    if (!candidate.type.startsWith('image/')) {
      setMessage({ tone: 'error', text: 'That file is not an image. Choose a PNG, JPG or TIFF scan.' })
      return
    }
    setFile(candidate)
    setMessage(null)
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    selectFile(e.dataTransfer.files?.[0])
  }

  function clearFile() {
    setFile(null)
    setMessage(null)
    if (inputRef.current) inputRef.current.value = ''
    inputRef.current?.focus()
  }

  const AlertToneIcon =
    message?.tone === 'error' || message?.tone === 'warning'
      ? AlertIcon
      : message?.tone === 'success'
        ? CheckCircleIcon
        : InfoIcon

  return (
    <section className="card upload-card" aria-labelledby="upload-heading">
      <div className="card__header">
        <div className="card__title-group">
          <span className="card__eyebrow">Step 1</span>
          <h2 id="upload-heading">Start a document review</h2>
          <p className="text-sm muted">
            Choose the document type first so structured fields receive the right checks.
          </p>
        </div>
      </div>

      <form className="stack" onSubmit={handleUpload}>
        {!file ? (
          <div
            className={`dropzone${dragging ? ' is-dragging' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <UploadIcon className="dropzone__icon" width={28} height={28} />
            <span className="dropzone__title">Drag a scan here, or browse files</span>
            <span className="text-sm muted" id="upload-hint">
              PNG, JPG or TIFF · one page per upload
            </span>
            <input
              ref={inputRef}
              id="document-file"
              className="dropzone__input"
              type="file"
              accept="image/*"
              aria-describedby="upload-hint"
              onChange={(e) => selectFile(e.target.files?.[0])}
            />
            <label className="sr-only" htmlFor="document-file">
              Choose a document image to upload
            </label>
          </div>
        ) : (
          <div className="file-preview">
            {thumbUrl ? (
              <img className="file-preview__thumb" src={thumbUrl} alt={`Preview of ${file.name}`} />
            ) : (
              <span className="file-preview__thumb" style={{ display: 'grid', placeItems: 'center' }}>
                <ImageIcon />
              </span>
            )}
            <span className="file-preview__meta">
              <span className="file-preview__name">{file.name}</span>
              <span className="text-xs muted">{formatBytes(file.size)}</span>
            </span>
            <button
              type="button"
              className="btn btn--ghost btn--icon"
              onClick={clearFile}
              disabled={busy}
            >
              <CloseIcon width={16} height={16} />
              <span className="sr-only">Remove {file.name}</span>
            </button>
          </div>
        )}

        <label className="field__label" htmlFor="document-type">
          Document type
          <select
            id="document-type"
            className="select"
            value={docType}
            onChange={(e) => setDocType(e.target.value)}
            disabled={busy}
          >
            <option value="generic">Generic government document</option>
            <option value="aadhaar">Aadhaar card</option>
            <option value="pan">PAN card</option>
            <option value="passport">Passport</option>
            <option value="marksheet">Mark sheet / certificate</option>
          </select>
          <span className="text-xs muted">
            This controls field validation and keeps the language model away from structured IDs
            and dates.
          </span>
        </label>

        <label className="field__label" htmlFor="ocr-languages">
          Document language
          <select
            id="ocr-languages"
            className="select"
            value={ocrLanguages}
            onChange={(e) => setOcrLanguages(e.target.value)}
            disabled={busy}
          >
            <option value="eng">English</option>
            <option value="asm">Assamese</option>
            <option value="ben">Bengali</option>
            <option value="guj">Gujarati</option>
            <option value="hin">Hindi</option>
            <option value="kan">Kannada</option>
            <option value="mal">Malayalam</option>
            <option value="mar">Marathi</option>
            <option value="ori">Odia</option>
            <option value="pan">Punjabi</option>
            <option value="san">Sanskrit</option>
            <option value="tam">Tamil</option>
            <option value="tel">Telugu</option>
            <option value="urd">Urdu</option>
            <option value="eng+hin">English + Hindi</option>
            <option value="eng+mar">English + Marathi</option>
            <option value="eng+tam">English + Tamil</option>
            <option value="eng+tel">English + Telugu</option>
          </select>
          <span className="text-xs muted">
            Choose the script used on the scan. Mixed selections support common English + native-language documents.
          </span>
        </label>

        <div className="btn-row">
          <button type="submit" className="btn btn--primary" disabled={busy || !file}>
            {busy && <span className="spinner" aria-hidden="true" />}
            {busy ? 'Processing…' : 'Upload & analyze'}
          </button>
          {pendingId && !busy && (
            <button type="button" className="btn" onClick={handleCheckAgain}>
              Check again
            </button>
          )}
        </div>

        {busy && (
          <div
            className="progress"
            role="progressbar"
            aria-label="Processing document"
            aria-valuetext="In progress"
          >
            <div className="progress__bar" />
          </div>
        )}

        <div aria-live="polite" aria-atomic="true">
          {message && (
            <p className={`alert alert--${message.tone}`}>
              <AlertToneIcon className="alert__icon" width={16} height={16} />
              <span>{message.text}</span>
            </p>
          )}
        </div>
      </form>
    </section>
  )
}
