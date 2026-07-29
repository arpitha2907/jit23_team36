import { useEffect, useMemo, useState } from 'react'
import {
  downloadDocumentImage,
  downloadDocumentText,
  getAudit,
  getPreviewImageUrl,
  getResult,
  submitCorrections,
} from '../api'
import {
  AlertIcon,
  CheckCircleIcon,
  CheckIcon,
  CloseIcon,
  DownloadIcon,
  FileTextIcon,
  HistoryIcon,
  ImageIcon,
  SearchIcon,
} from './Icons'

const FLAG_THRESHOLD = 0.4

function severityOf(score) {
  if (score == null) return 'low'
  if (score >= 0.7) return 'high'
  if (score >= FLAG_THRESHOLD) return 'mid'
  return 'low'
}

function ScoreMeter({ score }) {
  const pct = Math.round(Math.min(Math.max(score ?? 0, 0), 1) * 100)
  return (
    <span className="meter">
      <span className="meter__track" aria-hidden="true">
        <span
          className="meter__fill"
          data-severity={severityOf(score)}
          style={{ width: `${pct}%` }}
        />
      </span>
      <span className="meter__value">{score != null ? score.toFixed(2) : '—'}</span>
    </span>
  )
}

export default function Results({ result, onResult }) {
  const [decisions, setDecisions] = useState({})
  const [customInputs, setCustomInputs] = useState({})
  const [saved, setSaved] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [auditLog, setAuditLog] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [flaggedOnly, setFlaggedOnly] = useState(true)
  const [query, setQuery] = useState('')
  const [downloading, setDownloading] = useState('')

  const docId = result?.id

  // A new document means the previous document's decisions no longer apply.
  useEffect(() => {
    setDecisions({})
    setCustomInputs({})
    setSaved(false)
    setError('')
    setAuditLog(null)
    setFlaggedOnly(true)
    setQuery('')
  }, [docId])

  // Load the layout-reconstructed preview whenever a new result comes in,
  // and again after saving corrections (so it reflects the accepted text).
  useEffect(() => {
    if (!docId) return
    let cancelled = false
    let objectUrl = null
    getPreviewImageUrl(docId)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url)
          return
        }
        objectUrl = url
        setPreviewUrl(url)
      })
      .catch(() => setPreviewUrl(null))
    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [docId, saved])

  const fields = result?.fields ?? []

  const stats = useMemo(() => {
    const flagged = fields.filter((f) => (f.fused_error_score ?? 0) >= FLAG_THRESHOLD)
    const confidences = fields.map((f) => f.ocr_confidence).filter((c) => c != null)
    return {
      total: fields.length,
      flagged: flagged.length,
      clean: fields.length - flagged.length,
      avgConfidence: confidences.length
        ? confidences.reduce((a, b) => a + b, 0) / confidences.length
        : null,
    }
  }, [fields])

  const visibleFields = useMemo(() => {
    const q = query.trim().toLowerCase()
    return fields.filter((f) => {
      if (flaggedOnly && (f.fused_error_score ?? 0) < FLAG_THRESHOLD) return false
      if (!q) return true
      return Boolean(
        f.token?.toLowerCase().includes(q) || f.field_label?.toLowerCase().includes(q)
      )
    })
  }, [fields, flaggedOnly, query])

  if (!result) {
    return (
      <section className="card" aria-labelledby="results-empty-heading">
        <div className="empty-state">
          <FileTextIcon className="empty-state__icon" width={40} height={40} />
          <h2 id="results-empty-heading">No document loaded yet</h2>
          <p className="text-sm">
            Upload a scan above. Once the pipeline finishes, every recognized token appears here
            with its confidence, error score and suggested corrections.
          </p>
        </div>
      </section>
    )
  }

  const decisionCount = Object.keys(decisions).length

  function setDecision(fieldId, accept, correctedValue) {
    setDecisions((prev) => ({
      ...prev,
      [fieldId]: { field_id: fieldId, accept, corrected_value: correctedValue },
    }))
    setSaved(false)
  }

  function clearDecision(fieldId) {
    setDecisions((prev) => {
      const next = { ...prev }
      delete next[fieldId]
      return next
    })
    setSaved(false)
  }

  function saveBlob(blob, filename) {
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  async function handleDownload(kind) {
    setError('')
    setDownloading(kind)
    try {
      const type = result.doc_type || 'government-document'
      if (kind === 'image') {
        const blob = await downloadDocumentImage(result.id)
        saveBlob(blob, `${type}-corrected.png`)
      } else {
        const blob = await downloadDocumentText(result.id)
        saveBlob(blob, `${type}-corrected.txt`)
      }
    } catch (err) {
      setError('Download failed: ' + (err?.response?.data?.detail || err.message))
    } finally {
      setDownloading('')
    }
  }

  async function handleSave() {
    setError('')
    setSaving(true)
    try {
      await submitCorrections(result.id, Object.values(decisions))
      setSaved(true)

      // Refetch the whole document -- the /correct response doesn't include
      // per-field data, so this is the only way to get the table (OCR
      // confidence, error score, flagged badges) out of its pre-correction
      // state along with the corrected text/status and audit trail.
      const [updated, audit] = await Promise.all([getResult(result.id), getAudit(result.id)])
      onResult(updated)
      setAuditLog(audit)
    } catch (err) {
      if (err?.response?.status === 403) {
        setError(
          'Your account doesn\'t have permission to save corrections. ' +
            'Only "reviewer" or "admin" accounts can approve/reject — ' +
            'see the app README for how to upgrade your account role.'
        )
      } else {
        setError('Save failed: ' + (err?.response?.data?.detail || err.message))
      }
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="card results-card" aria-labelledby="results-heading">
      <div className="card__header">
        <div className="card__title-group">
          <span className="card__eyebrow">Step 2</span>
          <h2 id="results-heading">Review detected fields</h2>
          <p className="text-sm muted">
            Tokens with a fused error score of {FLAG_THRESHOLD.toFixed(1)} or higher are flagged as
            likely OCR errors. Nothing is written until you save.
          </p>
        </div>
        <div className="results-header__aside">
          <div className="results-header__meta">
            <span className="badge badge--brand">{result.doc_type || 'generic'}</span>
            <span className="badge">OCR: {result.ocr_languages || 'eng'}</span>
            {result.status && <span className="badge badge--ok">{result.status.replace('_', ' ')}</span>}
          </div>
          <div className="btn-row">
            <button
              type="button"
              className="btn btn--sm"
              onClick={() => handleDownload('image')}
              disabled={Boolean(downloading)}
            >
              <DownloadIcon width={15} height={15} />
              {downloading === 'image' ? 'Preparing image…' : 'Download image'}
            </button>
            <button
              type="button"
              className="btn btn--sm"
              onClick={() => handleDownload('text')}
              disabled={Boolean(downloading)}
            >
              <FileTextIcon width={15} height={15} />
              {downloading === 'text' ? 'Preparing text…' : 'Download text'}
            </button>
          </div>
        </div>
      </div>

      <div className="stat-grid" style={{ marginBottom: 'var(--space-5)' }}>
        <div className="stat">
          <div className="stat__value">{stats.total}</div>
          <div className="stat__label">Tokens detected</div>
        </div>
        <div className="stat stat--flag">
          <div className="stat__value">{stats.flagged}</div>
          <div className="stat__label">Flagged for review</div>
        </div>
        <div className="stat stat--ok">
          <div className="stat__value">{stats.clean}</div>
          <div className="stat__label">Passed all checks</div>
        </div>
        <div className="stat stat--brand">
          <div className="stat__value">
            {stats.avgConfidence != null ? `${stats.avgConfidence.toFixed(0)}%` : '—'}
          </div>
          <div className="stat__label">Mean OCR confidence</div>
        </div>
      </div>

      <div className="toolbar">
        <label className="switch" htmlFor="flagged-only">
          <input
            id="flagged-only"
            className="switch__input"
            type="checkbox"
            checked={flaggedOnly}
            onChange={(e) => setFlaggedOnly(e.target.checked)}
          />
          <span className="switch__track" aria-hidden="true" />
          <span>Show flagged fields only</span>
        </label>

        <div style={{ position: 'relative', flex: '1 1 240px', maxWidth: 320 }}>
          <label className="sr-only" htmlFor="field-search">
            Search tokens or field names
          </label>
          <SearchIcon
            width={16}
            height={16}
            style={{
              position: 'absolute',
              left: 12,
              top: '50%',
              transform: 'translateY(-50%)',
              color: 'var(--text-subtle)',
            }}
          />
          <input
            id="field-search"
            className="input"
            type="search"
            placeholder="Search tokens…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ paddingLeft: 36 }}
          />
        </div>
      </div>

      <p className="sr-only" aria-live="polite">
        Showing {visibleFields.length} of {stats.total} fields.
      </p>

      {visibleFields.length === 0 ? (
        <div className="empty-state">
          <CheckCircleIcon className="empty-state__icon" width={36} height={36} />
          <p>
            {query
              ? `No fields match “${query}”.`
              : 'Nothing flagged in this document — switch off the filter to see every token.'}
          </p>
        </div>
      ) : (
        <div className="table-scroll">
          <table className="table" role="table">
            <caption className="sr-only">
              Detected tokens with OCR confidence, fused error score, suggested corrections and your
              review decision.
            </caption>
            <thead role="rowgroup">
              <tr role="row">
                <th role="columnheader" scope="col">
                  Token
                </th>
                <th role="columnheader" scope="col">
                  OCR conf.
                </th>
                <th role="columnheader" scope="col">
                  Error score
                </th>
                <th role="columnheader" scope="col">
                  Suggestions
                </th>
                <th role="columnheader" scope="col">
                  Decision
                </th>
              </tr>
            </thead>
            <tbody role="rowgroup">
              {visibleFields.map((f) => {
                const flagged = (f.fused_error_score ?? 0) >= FLAG_THRESHOLD
                const decision = decisions[f.id]
                const rowClass = [
                  flagged ? 'is-flagged' : '',
                  decision?.accept === true ? 'is-accepted' : '',
                  decision?.accept === false ? 'is-rejected' : '',
                ]
                  .filter(Boolean)
                  .join(' ')

                return (
                  <tr key={f.id} role="row" className={rowClass}>
                    <td role="cell" data-label="Token">
                      <span className="token-cell">
                        <span className="token-cell__value">{f.token}</span>
                        {f.field_label && (
                          <span className="badge badge--brand">{f.field_label}</span>
                        )}
                        {f.is_label_token && <span className="badge">Field label</span>}
                        {flagged && <span className="badge badge--flag">Flagged</span>}
                      </span>
                    </td>

                    <td role="cell" data-label="OCR conf.">
                      <span className="mono text-sm">
                        {f.ocr_confidence != null ? `${f.ocr_confidence.toFixed(1)}%` : '—'}
                      </span>
                    </td>

                    <td role="cell" data-label="Error score">
                      <ScoreMeter score={f.fused_error_score} />
                    </td>

                    <td role="cell" data-label="Suggestions">
                      {f.suggestions.length > 0 ? (
                        <span className="btn-row">
                          {f.suggestions.map((s) => (
                            <button
                              key={s}
                              type="button"
                              className="btn btn--sm chip-btn"
                              aria-pressed={
                                decision?.accept === true && decision.corrected_value === s
                              }
                              aria-label={`Replace “${f.token}” with “${s}”`}
                              onClick={() => setDecision(f.id, true, s)}
                            >
                              {s}
                            </button>
                          ))}
                        </span>
                      ) : (
                        <span className="text-xs muted">No automated suggestion</span>
                      )}
                    </td>

                    <td role="cell" data-label="Decision">
                      <span className="decision-cell">
                        <span className="btn-row">
                          <button
                            type="button"
                            className="btn btn--sm btn--accept"
                            aria-pressed={decision?.accept === true}
                            aria-label={`Accept correction for “${f.token}”`}
                            onClick={() =>
                              setDecision(f.id, true, decision?.corrected_value || f.token)
                            }
                          >
                            <CheckIcon width={14} height={14} />
                            Accept
                          </button>
                          <button
                            type="button"
                            className="btn btn--sm btn--reject"
                            aria-pressed={decision?.accept === false}
                            aria-label={`Reject correction for “${f.token}”`}
                            onClick={() => setDecision(f.id, false, null)}
                          >
                            <CloseIcon width={14} height={14} />
                            Reject
                          </button>
                          {decision && (
                            <button
                              type="button"
                              className="btn btn--sm btn--ghost"
                              onClick={() => clearDecision(f.id)}
                            >
                              Clear
                              <span className="sr-only"> decision for “{f.token}”</span>
                            </button>
                          )}
                        </span>

                        <span className="decision-cell__custom">
                          <label className="sr-only" htmlFor={`custom-${f.id}`}>
                            Custom value for “{f.token}”
                          </label>
                          <input
                            id={`custom-${f.id}`}
                            className="input input--inline"
                            type="text"
                            placeholder="Type correct value…"
                            value={customInputs[f.id] ?? ''}
                            onChange={(e) =>
                              setCustomInputs((prev) => ({ ...prev, [f.id]: e.target.value }))
                            }
                          />
                          <button
                            type="button"
                            className="btn btn--sm"
                            disabled={!customInputs[f.id]}
                            onClick={() => setDecision(f.id, true, customInputs[f.id] || '')}
                          >
                            Use
                            <span className="sr-only"> this value for “{f.token}”</span>
                          </button>
                        </span>

                        {decision && (
                          <span className="text-xs muted">
                            {decision.accept
                              ? `Will save as “${decision.corrected_value}”`
                              : 'Will keep the original OCR text'}
                          </span>
                        )}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="save-bar">
        <p className="text-sm muted">
          {decisionCount === 0
            ? 'No decisions recorded yet — accept, reject or type a replacement to get started.'
            : `${decisionCount} field${decisionCount === 1 ? '' : 's'} decided.`}
        </p>
        <button
          type="button"
          className="btn btn--primary"
          onClick={handleSave}
          disabled={saving || decisionCount === 0}
        >
          {saving && <span className="spinner" aria-hidden="true" />}
          {saving ? 'Saving…' : 'Save corrections'}
        </button>
      </div>

      <div aria-live="polite">
        {saved && !error && (
          <p className="alert alert--success">
            <CheckCircleIcon className="alert__icon" width={16} height={16} />
            <span>Corrections saved and written to the audit log.</span>
          </p>
        )}
        {error && (
          <p className="alert alert--error" role="alert">
            <AlertIcon className="alert__icon" width={16} height={16} />
            <span>{error}</span>
          </p>
        )}
      </div>

      {result.corrected_text != null && (
        <div className="card__section stack">
          <div className="card__header" style={{ marginBottom: 0 }}>
            <h3>
              <FileTextIcon width={16} height={16} /> Corrected document
            </h3>
            <span className="text-xs muted">Ready to download as plain text</span>
          </div>
          <p className="prose-box">{result.corrected_text}</p>
        </div>
      )}

      {previewUrl && (
        <div className="card__section stack">
          <h3>
            <ImageIcon width={16} height={16} /> Layout preview
          </h3>
          <p className="text-sm muted">
            Approved corrections are placed over the original scan at the same coordinates. The
            renderer preserves the document layout while replacing only the reviewed words.
          </p>
          <figure className="preview-figure">
            <img
              src={previewUrl}
              alt="Corrected document text redrawn in the original scan layout"
            />
          </figure>
        </div>
      )}

      {auditLog && auditLog.length > 0 && (
        <div className="card__section stack">
          <h3>
            <HistoryIcon width={16} height={16} /> Audit trail
          </h3>
          <div className="table-scroll">
            <table className="table" role="table">
              <caption className="sr-only">Recorded actions for this document.</caption>
              <thead role="rowgroup">
                <tr role="row">
                  <th role="columnheader" scope="col">
                    Action
                  </th>
                  <th role="columnheader" scope="col">
                    Detail
                  </th>
                  <th role="columnheader" scope="col">
                    When
                  </th>
                </tr>
              </thead>
              <tbody role="rowgroup">
                {auditLog.map((a, i) => (
                  <tr role="row" key={`${a.timestamp}-${i}`}>
                    <td role="cell" data-label="Action">
                      <span className="badge">{a.action}</span>
                    </td>
                    <td role="cell" data-label="Detail">
                      {a.detail}
                    </td>
                    <td role="cell" data-label="When">
                      <time dateTime={a.timestamp}>{new Date(a.timestamp).toLocaleString()}</time>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  )
}
