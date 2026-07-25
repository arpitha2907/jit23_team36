import { useEffect, useState } from 'react'
import { getAudit, getPreviewImageUrl, getResult, submitCorrections } from '../api'

export default function Results({ result }) {
  const [decisions, setDecisions] = useState({})
  const [customInputs, setCustomInputs] = useState({})
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')
  const [correctedText, setCorrectedText] = useState(null)
  const [docStatus, setDocStatus] = useState(null)
  const [auditLog, setAuditLog] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)

  // Load the layout-reconstructed preview whenever a new result comes in,
  // and again after saving corrections (so it reflects the accepted text).
  useEffect(() => {
    if (!result) return
    let cancelled = false
    let objectUrl = null
    getPreviewImageUrl(result.id)
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
  }, [result, saved])

  if (!result) return null

  function setDecision(fieldId, accept, correctedValue) {
    setDecisions((prev) => ({
      ...prev,
      [fieldId]: { field_id: fieldId, accept, corrected_value: correctedValue },
    }))
    setSaved(false)
  }

  async function handleSave() {
    setError('')
    try {
      await submitCorrections(result.id, Object.values(decisions))
      setSaved(true)

      // Pull the freshly updated corrected text/status and audit trail so
      // they're visible right here instead of only via the API docs.
      const [updated, audit] = await Promise.all([
        getResult(result.id),
        getAudit(result.id),
      ])
      setCorrectedText(updated.corrected_text)
      setDocStatus(updated.status)
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
    }
  }

  return (
    <div className="card">
      <h2>Detected fields</h2>
      <p>Rows highlighted in orange were flagged as likely OCR errors (fused error score ≥ 0.4). Where a row is recognized as the value for a known field (e.g. Date of Birth, ID Number), that field name is shown under the token.</p>
      <table>
        <thead>
          <tr>
            <th>Token</th>
            <th>OCR conf.</th>
            <th>Error score</th>
            <th>Suggestions</th>
            <th>Decision</th>
          </tr>
        </thead>
        <tbody>
          {result.fields.map((f) => {
            const flagged = (f.fused_error_score ?? 0) >= 0.4
            const decision = decisions[f.id]
            return (
              <tr key={f.id} className={flagged ? 'flagged' : ''}>
                <td>
                  {f.token}
                  {f.is_label_token && (
                    <div style={{ fontSize: 11, color: '#888' }}>(field label)</div>
                  )}
                  {f.field_label && (
                    <div style={{ fontSize: 11, color: '#1c4ed8' }}>Field: {f.field_label}</div>
                  )}
                </td>
                <td>{f.ocr_confidence != null ? f.ocr_confidence.toFixed(1) : '-'}</td>
                <td>{f.fused_error_score != null ? f.fused_error_score.toFixed(2) : '-'}</td>
                <td>
                  {f.suggestions.map((s) => (
                    <button key={s} type="button" onClick={() => setDecision(f.id, true, s)}>
                      {s}
                    </button>
                  ))}
                  {f.suggestions.length === 0 && (
                    <span style={{ color: '#888', fontSize: 12 }}>No automated suggestion</span>
                  )}
                </td>
                <td>
                  <button type="button" onClick={() => setDecision(f.id, true, decision?.corrected_value || f.token)}>
                    Accept{decision?.accept === true ? ' ✓' : ''}
                  </button>
                  <button
                    type="button"
                    onClick={() => setDecision(f.id, false, null)}
                    style={{ background: decision?.accept === false ? '#b00020' : '#999' }}
                  >
                    Reject{decision?.accept === false ? ' ✓' : ''}
                  </button>

                  <div style={{ marginTop: 6, display: 'flex', gap: 4 }}>
                    <input
                      type="text"
                      placeholder="Type correct value..."
                      value={customInputs[f.id] ?? ''}
                      onChange={(e) =>
                        setCustomInputs((prev) => ({ ...prev, [f.id]: e.target.value }))
                      }
                      style={{ width: 140, padding: 4, margin: 0 }}
                    />
                    <button
                      type="button"
                      onClick={() => setDecision(f.id, true, customInputs[f.id] || '')}
                      disabled={!customInputs[f.id]}
                      style={{ padding: '4px 8px' }}
                    >
                      Use this
                    </button>
                  </div>

                  {decision && (
                    <div style={{ fontSize: 12, color: '#555', marginTop: 4 }}>
                      {decision.accept
                        ? `Will save as: "${decision.corrected_value}"`
                        : 'Will keep original OCR text (no correction applied)'}
                    </div>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <button type="button" onClick={handleSave}>
        Save corrections
      </button>
      {saved && <p>Saved.</p>}
      {error && <p className="error">{error}</p>}

      {correctedText !== null && (
        <div style={{ marginTop: 20 }}>
          <h3>Corrected document {docStatus && `(status: ${docStatus})`}</h3>
          <p style={{ background: '#f4f5f7', padding: 10, borderRadius: 4 }}>{correctedText}</p>
        </div>
      )}

      {previewUrl && (
        <div style={{ marginTop: 20 }}>
          <h3>Layout preview</h3>
          <p style={{ fontSize: 13, color: '#555' }}>
            Corrected text redrawn at each word's original position from the scan —
            labels shown in gray/bold, values in black, mirroring the input document's layout.
          </p>
          <img
            src={previewUrl}
            alt="Layout-reconstructed document preview"
            style={{ maxWidth: '100%', border: '1px solid #ddd', borderRadius: 4 }}
          />
        </div>
      )}

      {auditLog && auditLog.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3>Audit trail</h3>
          <table>
            <thead>
              <tr>
                <th>Action</th>
                <th>Detail</th>
                <th>When</th>
              </tr>
            </thead>
            <tbody>
              {auditLog.map((a, i) => (
                <tr key={i}>
                  <td>{a.action}</td>
                  <td>{a.detail}</td>
                  <td>{new Date(a.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
