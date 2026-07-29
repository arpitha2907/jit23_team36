import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const client = axios.create({ baseURL: API_BASE })

client.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Lets App.jsx react to session expiry (expired/invalid token) without every
// call site needing to special-case 401s itself.
let onUnauthorized = null

export function setUnauthorizedHandler(fn) {
  onUnauthorized = fn
}

client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.response?.status === 401) {
      localStorage.removeItem('token')
      onUnauthorized?.()
    }
    return Promise.reject(error)
  }
)

export async function login(email, password) {
  const form = new URLSearchParams()
  form.append('username', email)
  form.append('password', password)
  const { data } = await client.post('/auth/login', form, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  })
  return data
}

export async function register(email, password, role = 'uploader') {
  const { data } = await client.post('/auth/register', { email, password, role })
  return data
}

export async function uploadDocument(file, docType = 'generic', ocrLanguages = 'eng') {
  const form = new FormData()
  form.append('file', file)
  form.append('doc_type', docType)
  form.append('ocr_languages', ocrLanguages)
  const { data } = await client.post('/documents/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function getStatus(id) {
  const { data } = await client.get(`/documents/${id}/status`)
  return data
}

export async function getResult(id) {
  const { data } = await client.get(`/documents/${id}/result`)
  return data
}

export async function submitCorrections(id, decisions) {
  const { data } = await client.post(`/documents/${id}/correct`, { decisions })
  return data
}

export async function getAudit(id) {
  const { data } = await client.get(`/documents/${id}/audit`)
  return data
}

// Fetched as a blob (not a plain <img src="...">) because the endpoint
// requires the same JWT auth header as everything else, which a bare <img>
// tag can't send.
export async function getPreviewImageUrl(id) {
  const response = await client.get(`/documents/${id}/preview-image`, {
    responseType: 'blob',
  })
  return URL.createObjectURL(response.data)
}

export async function downloadDocumentImage(id) {
  const { data } = await client.get(`/documents/${id}/download/image`, {
    responseType: 'blob',
  })
  return data
}

export async function downloadDocumentText(id) {
  const { data } = await client.get(`/documents/${id}/download/text`, {
    responseType: 'blob',
  })
  return data
}

export default client
