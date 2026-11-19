// Data access for the probe export (probe_teachers.py export) and audio streaming.
// Data files live in public/data; audio streams from the Music Lenses Hugging Face Space (no audio bytes in this repo).

export const DATA = `${import.meta.env.BASE_URL}data/`
export const AUDIO = id => `https://pliploop-music-lenses-audio-stream.hf.space/audio/${id}.mp3`

export const MODELS = {
  qwen3emb8b_q: ['Qwen3-Emb 8B', 'question wording'],
  inbedder7b: ['InBedder 7B', 'answer-is-embedding'],
  inbedder7b_na: ['InBedder 7B · none', 'may answer “none”'],
  llm_extract: ['LLM extraction', 'Qwen3.5-9B → Qwen3-Emb'],
}
export const NA_MODES = [
  ['off', 'Off', 'Show every pair'],
  ['jev', 'JEV-9B', 'Calibrated P(answerable) below 0.5'],
  ['anchor', 'Embedder confidence', 'Too close to the null caption “A piece of music.”'],
  ['none', 'InBedder says “none”', 'Literal answer of InBedder 7B · none'],
  ['llm', 'LLM says “not mentioned”', 'Qwen3.5-9B extraction answer'],
]
export const KIND = { attribute: 'Attribute', question: 'Yes / no question', paraphrase: 'Paraphrase' }
export const kindOf = k => KIND[k.split(':')[0]] || k
// Qwen3.5-9B often paraphrases the requested "not mentioned"
export const NOT_MENTIONED = /^not mentioned|\b(does not|doesn't|do not) (state|specify|mention|imply|indicate|say)|\bnot (specified|stated|mentioned|indicated)/i

// InBedder is trained for ~3-token answers; greedy decoding past that loops, so cut at the first repeated 3-word phrase
export function trimLoop(t) {
  const w = t.split(/\s+/), seen = new Set()
  for (let i = 0; i + 2 < w.length; i++) {
    const g = w.slice(i, i + 3).join(' ').toLowerCase()
    if (seen.has(g)) return w.slice(0, i).join(' ') + ' …'
    seen.add(g)
  }
  return t
}

function b64(t) {
  const s = atob(t.trim()), u = new Uint8Array(s.length)
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i)
  return u.buffer
}

const jsonCache = {}
export async function getJson(path) {
  if (!(path in jsonCache)) jsonCache[path] = fetch(DATA + path).then(r => (r.ok ? r.json() : null)).catch(() => null)
  return jsonCache[path]
}

export async function getJev(meta) {
  if (!meta.jev_order) return null
  try { return new Uint8Array(b64(await (await fetch(DATA + 'bin/jev.txt')).text())) } catch { return null }
}

// One (teacher, instruction) file: int8 compressed embeddings, exact top-K neighbours, null-caption similarity
const fileCache = {}
export function getFile(meta, model, iid) {
  const key = `${model}/${iid}`
  if (!fileCache[key]) {
    fileCache[key] = fetch(`${DATA}bin/${model}__${iid}.txt`).then(r => r.text()).then(t => {
      const buf = b64(t), info = meta.files[key], { n: N, k: K, dim: D } = meta
      let o = 0
      const take = (T, len, bytes) => { const a = new T(buf, o, len); o += len * bytes; return a }
      const f = { info, emb: take(Int8Array, N * D, 1), idx: take(Uint16Array, N * K, 2), sim: take(Uint16Array, N * K, 2) }
      if (iid !== 'none' && info.app) f.app = take(Uint16Array, N, 2)
      f.sc2 = Float32Array.from(info.scale, s => s * s)
      return f
    })
  }
  return fileCache[key]
}

export const deq = (q, lo, hi) => lo + ((hi - lo) * q) / 65535
export function cell(f, dim, a, b) {
  let s = 0
  const e = f.emb, oa = a * dim, ob = b * dim
  for (let k = 0; k < dim; k++) s += e[oa + k] * e[ob + k] * f.sc2[k]
  return s
}

export function search(C, q) {
  const toks = q.toLowerCase().split(/\s+/).filter(Boolean), hits = []
  for (let i = 0; i < C.length && hits.length < 30; i++) {
    const c = C[i], hay = `${c.caption} ${c.genres.join(' ')} #${i + 1}`.toLowerCase()
    if (toks.every(t => hay.includes(t))) hits.push(i)
  }
  return hits
}

export const WHITE = [255, 255, 255], RED = [196, 36, 48], GREEN = [16, 128, 78]
export const mix = (a, b, t) => a.map((x, k) => Math.round(x + (b[k] - x) * t))
