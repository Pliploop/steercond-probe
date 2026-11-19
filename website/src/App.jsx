import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { AUDIO, GREEN, MODELS, NA_MODES, NOT_MENTIONED, RED, WHITE, cell, deq, getFile, getJev, getJson, kindOf, mix, search, trimLoop } from './lib.js'

const START_INSTR = 'instrumentation'
// The audio Space only allows cross-origin reads from the project site; elsewhere play without the live waveform.
const LIVE_WAVE = typeof window !== 'undefined' && window.location.origin === 'https://pliploop.github.io'

/* ---------- audio: one shared element, full 30 s Music4All clips streamed from Hugging Face ---------- */
function useAudio() {
  const ref = useRef(null)
  const [s, setS] = useState({ idx: null, playing: false, loading: false, t: 0, dur: 30, error: false })
  function el() {
    if (ref.current) return ref.current
    const audio = new Audio()
    audio.preload = 'none'
    if (LIVE_WAVE) audio.crossOrigin = 'anonymous'
    audio.addEventListener('timeupdate', () => setS(p => ({ ...p, t: audio.currentTime })))
    audio.addEventListener('durationchange', () => setS(p => ({ ...p, dur: audio.duration || 30 })))
    audio.addEventListener('playing', () => setS(p => ({ ...p, playing: true, loading: false })))
    audio.addEventListener('pause', () => setS(p => ({ ...p, playing: false })))
    audio.addEventListener('ended', () => setS(p => ({ ...p, playing: false, t: 0 })))
    audio.addEventListener('error', () => setS(p => ({ ...p, playing: false, loading: false, error: true })))
    ref.current = { audio, ctx: null, analyser: null }
    return ref.current
  }
  function wire(r) {  // Web Audio analyser for the live waveform; needs a user gesture and CORS
    if (!LIVE_WAVE || r.ctx) return
    try {
      r.ctx = new AudioContext()
      const src = r.ctx.createMediaElementSource(r.audio)
      r.analyser = r.ctx.createAnalyser()
      r.analyser.fftSize = 1024
      src.connect(r.analyser).connect(r.ctx.destination)
    } catch { r.analyser = null }
  }
  function toggle(idx, id) {
    const r = el()
    wire(r)
    r.ctx?.resume()
    if (s.idx === idx) { r.audio.paused ? r.audio.play() : r.audio.pause(); return }
    r.audio.src = AUDIO(id)
    setS({ idx, playing: false, loading: true, t: 0, dur: 30, error: false })
    r.audio.play().catch(() => setS(p => ({ ...p, loading: false, error: true })))
  }
  function stop() { const r = el(); r.audio.pause(); r.audio.removeAttribute('src'); setS({ idx: null, playing: false, loading: false, t: 0, dur: 30, error: false }) }
  function seek(f) { const r = el(); if (r.audio.duration) r.audio.currentTime = f * r.audio.duration }
  return { ...s, toggle, stop, seek, analyser: () => ref.current?.analyser }
}

function Wave({ audio }) {
  const cv = useRef(null)
  useEffect(() => {
    const a = audio.analyser()
    if (!a || !audio.playing) return
    const ctx = cv.current.getContext('2d'), buf = new Uint8Array(a.fftSize)
    const slow = matchMedia('(prefers-reduced-motion: reduce)').matches
    let raf, last = 0
    const draw = ts => {
      raf = requestAnimationFrame(draw)
      if (slow && ts - last < 100) return
      last = ts
      a.getByteTimeDomainData(buf)
      const { width: w, height: h } = cv.current
      ctx.clearRect(0, 0, w, h)
      ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,255,255,.9)'; ctx.lineJoin = 'round'
      ctx.beginPath()
      for (let x = 0; x < w; x++) {
        const v = (buf[Math.floor((x / w) * buf.length)] - 128) / 128
        const y = h / 2 + v * h * 0.9
        x ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
      }
      ctx.stroke()
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [audio.playing, audio.idx])
  return <canvas ref={cv} width={120} height={28} className="wave" aria-hidden="true" />
}

/* ---------- small pieces ---------- */
const genresOf = c => c.genres.slice(0, 3).join(' · ')
function Meta({ C, i }) {
  const c = C[i]
  return (
    <span className="meta">#{i + 1} · {genresOf(c)}
      {c.sp && <> · <a className="sp" href={`https://open.spotify.com/track/${c.sp}`} target="_blank" rel="noopener noreferrer">Spotify ↗</a></>}
    </span>
  )
}
function Play({ audio, i, C }) {
  const on = audio.idx === i && (audio.playing || audio.loading)
  return (
    <button className={`play ${on ? 'on' : ''}`} aria-label={`${on ? 'Pause' : 'Play'} track #${i + 1}`}
      onClick={e => { e.stopPropagation(); audio.toggle(i, C[i].id) }}>
      {audio.idx === i && audio.loading ? <span className="spin" /> : on ? '❚❚' : '▶'}
    </button>
  )
}
function Seg({ value, options, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([v, t]) => <button key={v} aria-pressed={value === v} onClick={() => onChange(v)}>{t}</button>)}
    </div>
  )
}

/* ---------- app ---------- */
export default function App() {
  const [D, setD] = useState(null)  // { M, C, JEV }
  const [err, setErr] = useState(null)
  const [model, setModel] = useState(null)
  const [instr, setInstr] = useState(START_INSTR)
  const [view, setView] = useState('after')
  const [topView, setTopView] = useState('after')
  const [na, setNa] = useState('off')
  const [thr, setThr] = useState(null)
  const [slots, setSlots] = useState([])
  const [hover, setHover] = useState(null)
  const [pin, setPin] = useState(null)
  const [tab, setTab] = useState('rank')
  const [F, setF] = useState(null)  // { f0, f1, gens, llm } for (model, instr)
  const [drawer, setDrawer] = useState(false)
  const audio = useAudio()

  useEffect(() => {
    Promise.all([getJson('meta.json'), getJson('captions.json')]).then(async ([M, C]) => {
      if (!M || !C) throw new Error('meta.json or captions.json is missing')
      setD({ M, C, JEV: await getJev(M) })
      setModel(M.models.includes('inbedder7b') ? 'inbedder7b' : M.models[0])
      setSlots([0, 1, 2, 3, 4, 5, 6, 7].map(k => (k * 1237 + 11) % M.n))
    }).catch(e => setErr(e.message))
  }, [])

  useEffect(() => {
    if (!D || !model) return
    let live = true
    const { M } = D
    const want = (m, i) => (M.gens[m] || []).includes(i) ? getJson(`gen/${m}/${i}.json`) : null
    Promise.all([getFile(M, model, 'none'), getFile(M, model, instr), M.llm.includes(instr) ? getJson(`llm/${instr}.json`) : null,
      want('inbedder7b', instr), want('inbedder7b_na', instr)])
      .then(([f0, f1, llm, g, gna]) => { if (live) { setF({ f0, f1, llm, gens: { inbedder7b: g, inbedder7b_na: gna }, key: `${model}/${instr}` }); setPin(null) } })
      .catch(e => setErr(e.message))
    return () => { live = false }
  }, [D, model, instr])

  const M = D?.M, C = D?.C, N = M?.n, K = M?.k, DIM = M?.dim
  const ready = D && F && F.key === `${model}/${instr}`

  /* N/A */
  const jevRow = useMemo(() => (D?.JEV && M.jev_order ? M.jev_order.indexOf(instr) : -1), [D, instr])
  const jevP = i => (jevRow < 0 ? null : D.JEV[jevRow * N + i] / 255)
  const appSim = i => (F?.f1?.app ? deq(F.f1.app[i], -1, 1) : null)
  useEffect(() => {  // default confidence threshold: match JEV's N/A rate for this instruction, else the top 10 %
    if (!ready || !F.f1.app) return setThr(null)
    const v = Array.from(F.f1.app, q => deq(q, -1, 1)).sort((x, y) => y - x)
    let rate = 0.1
    if (jevRow >= 0) { let c = 0; for (let i = 0; i < N; i++) c += D.JEV[jevRow * N + i] < 128; rate = c / N }
    setThr(v[Math.min(N - 1, Math.max(0, Math.round(rate * N) - 1))])
  }, [ready, F, jevRow])
  const isNA = i => {
    if (!ready || na === 'off') return false
    if (na === 'jev') { const p = jevP(i); return p !== null && p < 0.5 }
    if (na === 'anchor') { const a = appSim(i); return a !== null && thr !== null && a >= thr }
    if (na === 'none') return !!F.gens.inbedder7b_na && /^none\b/i.test(F.gens.inbedder7b_na[i])
    if (na === 'llm') return !!F.llm && NOT_MENTIONED.test(F.llm[i])
    return false
  }
  const naCount = useMemo(() => { if (!ready || na === 'off') return 0; let c = 0; for (let i = 0; i < N; i++) c += isNA(i); return c }, [ready, na, thr, F])
  const avail = D ? { jev: !!D.JEV, anchor: Object.values(M.files).some(f => f.app), none: !!(M.gens.inbedder7b_na || []).length, llm: !!M.llm.length } : {}

  if (err) return <div className="wrap"><Header status={`Could not load data: ${err}`} /></div>
  if (!D) return <div className="wrap"><Header status="Loading data…" /><div className="card skeleton" /></div>

  const I = M.instructions.find(x => x[0] === instr)
  const order = M.instructions.map(x => x[0]).filter(x => x !== 'none')
  const step = d => setInstr(order[(order.indexOf(instr) + d + order.length) % order.length])
  const info = key => M.files[`${model}/${key}`]
  const sim = (f, a, b) => cell(f, DIM, a, b)

  function facts(i) {
    if (!ready) return null
    const p = jevP(i), a = appSim(i), g = model.startsWith('inbedder') ? F.gens[model] : null
    return (
      <>
        {isNA(i) && <span className="tagna">Not applicable</span>}
        <div className="facts">
          {g && <div>InBedder answer · <b>{trimLoop(g[i])}</b></div>}
          {F.llm && <div>LLM answer · <b>{F.llm[i]}</b></div>}
          {(p !== null || a !== null) && <div>{p !== null && <>JEV P(answerable) <b className="num">{p.toFixed(2)}</b></>}{p !== null && a !== null && ' · '}{a !== null && <>null-caption sim <b className="num">{a.toFixed(2)}</b></>}</div>}
        </div>
      </>
    )
  }

  return (
    <div className="wrap">
      <Header status={ready ? `${N.toLocaleString()} Music4All captions · ${M.instructions.length - 1} instructions · ${M.models.length} teachers` : `Loading ${MODELS[model]?.[0] || model}…`} />

      <section className="card controls" aria-label="Controls">
        <div className="ctl"><span className="label">Embedder</span>
          <div className="pills">
            {M.models.map(k => (
              <button key={k} className="pill" aria-pressed={k === model} onClick={() => setModel(k)}>
                <b>{MODELS[k]?.[0] || k}</b><small>{MODELS[k]?.[1]}</small>
              </button>
            ))}
          </div>
        </div>
        <div className="ctl"><label className="label" htmlFor="instr">Modulating instruction</label>
          <div className="stepper">
            <button className="iconbtn" aria-label="Previous instruction" onClick={() => step(-1)}>‹</button>
            <select id="instr" value={instr} onChange={e => setInstr(e.target.value)}>
              {Object.entries(M.instructions.filter(x => x[0] !== 'none').reduce((g, x) => ((g[kindOf(x[3])] ||= []).push(x), g), {})).map(([g, xs]) => (
                <optgroup key={g} label={g}>{xs.map(([iid, s]) => <option key={iid} value={iid}>{s}</option>)}</optgroup>
              ))}
            </select>
            <button className="iconbtn" aria-label="Next instruction" onClick={() => step(1)}>›</button>
          </div>
        </div>
        <NaDrawer {...{ open: drawer, setOpen: setDrawer, na, setNa, avail, naCount, thr, setThr, fi: info(instr), hasApp: !!F?.f1?.app && ready }} />
      </section>

      <section className="headline" aria-live="polite">
        <span className="label">Instruction</span>
        <h2>{I[1]}</h2>
        <p className="q"><span className="kind">{kindOf(I[3])}</span>{I[2]}</p>
      </section>

      <section className="card stage" aria-label="Similarity matrix">
        <div className="stage-head">
          <Seg label="Matrix view" value={view} onChange={setView} options={[['before', 'No instruction'], ['after', 'With instruction'], ['delta', 'Change']]} />
          {ready && <Legend view={view} fi={info(view === 'before' ? 'none' : instr)} N={N} />}
        </div>
        <div className="scroller">
          <div className="board">
            <Slots {...{ C, slots, setSlots, act: pin || hover, audio, setPin }} />
            {ready ? <Grid {...{ slots, view, F, sim, isNA, info, instr, pin, setPin, setHover }} /> : <div className="grid-wait">Loading…</div>}
          </div>
        </div>
        <div className="slot-actions">
          <button className="ghost" onClick={() => setSlots([...slots, Math.floor(Math.random() * N)])}>+ Add caption</button>
          <button className="ghost" onClick={() => { setSlots(slots.map(() => Math.floor(Math.random() * N))); setPin(null) }}>Shuffle all</button>
          <span className="hint">Click a caption to search · ▶ streams the 30 s clip · hover a cell to read both, click to pin</span>
        </div>
      </section>

      {ready && <Lists {...{ C, M, K, N, F, slots, pin, instr, I, tab, setTab, topView, setTopView, sim, info, facts, audio }} />}

      {ready && <Panels {...{ C, slots, act: pin || hover, pinned: !!pin, unpin: () => setPin(null), F, sim, instr, facts, audio }} />}
      {audio.idx !== null && <NowPlaying {...{ audio, C }} />}
    </div>
  )
}

function Header({ status }) {
  return (
    <header className="top">
      <span className="mark" aria-hidden="true" />
      <div><h1>Conditional Similarity Probe</h1><p id="status">{status}</p></div>
    </header>
  )
}

function Legend({ view, fi, N }) {
  if (!fi) return null
  if (view === 'delta') {
    const m = Math.max(Math.abs(fi.dlo), Math.abs(fi.dhi))
    return <div className="legend"><span className="num">−{m.toFixed(2)}</span><i style={{ background: `linear-gradient(90deg,rgb(${RED}),#fff,rgb(${GREEN}))` }} /><span className="num">+{m.toFixed(2)}</span><span>change from adding the instruction</span></div>
  }
  return <div className="legend"><span className="num">{fi.lo.toFixed(2)}</span><i style={{ background: `linear-gradient(90deg,#fff,rgb(${RED}))` }} /><span className="num">{fi.hi.toFixed(2)}</span><span>cosine, min–max over all {N.toLocaleString()} captions</span></div>
}

function NaDrawer({ open, setOpen, na, setNa, avail, naCount, thr, setThr, fi, hasApp }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!open) return
    const close = e => { if (!ref.current?.contains(e.target)) setOpen(false) }
    document.addEventListener('click', close)
    return () => document.removeEventListener('click', close)
  }, [open])
  const fmt = x => (x == null ? '–' : x.toFixed(2))
  return (
    <div className="drawer" ref={ref}>
      <button className={`drawer-btn ${open ? 'open' : ''}`} aria-expanded={open} onClick={() => setOpen(!open)}>
        Not applicable <span className={`badge ${na !== 'off' ? 'on' : ''}`}>{na === 'off' ? 'Off' : `${naCount.toLocaleString()} N/A`}</span>
      </button>
      {open && (
        <div className="drawer-body">
          <div className="ctl"><span className="label">Mark captions that cannot answer the instruction</span>
            <div className="radios">
              {NA_MODES.map(([v, t, s]) => {
                const off = v !== 'off' && !avail[v]
                return (
                  <label key={v} className={`radio ${off ? 'off' : ''}`}>
                    <input type="radio" name="na" value={v} checked={na === v} disabled={off} onChange={() => setNa(v)} />
                    <span><b>{t}</b><small>{off ? 'Not computed for this data' : s}</small></span>
                  </label>
                )
              })}
            </div>
          </div>
          {na === 'anchor' && hasApp && thr !== null && (
            <div className="slider"><span className="label">Confidence threshold</span>
              <input type="range" min={fi.anchor_lo} max={fi.anchor_hi} step="0.001" value={thr} onChange={e => setThr(+e.target.value)} />
              <span className="meta">similarity ≥ {thr.toFixed(3)} counts as N/A · default matches JEV's N/A rate</span>
            </div>
          )}
          <div className="ctl"><span className="label">N/A detection, this embedder and instruction</span>
            <div className="stats">
              <div className="stat"><b>{na === 'off' ? '–' : naCount.toLocaleString()}</b><span>captions marked N/A</span></div>
              <div className="stat"><b>{fmt(fi?.auroc_anchor_vs_jev)}</b><span>confidence vs JEV, AUROC</span></div>
              <div className="stat"><b>{fmt(fi?.auroc_anchor_vs_llm)}</b><span>confidence vs LLM, AUROC</span></div>
              <div className="stat"><b>{fmt(fi?.auroc_jev_vs_llm)}</b><span>JEV vs LLM, AUROC</span></div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function Slots({ C, slots, setSlots, act, audio, setPin }) {
  return (
    <div className="slots">
      {slots.map((idx, r) => <Slot key={r} {...{ C, idx, r, n: slots.length, act, audio, set: i => setSlots(slots.map((x, k) => (k === r ? i : x))),
        remove: () => { setSlots(slots.filter((_, k) => k !== r)); setPin(null) } }} />)}
    </div>
  )
}

function Slot({ C, idx, r, n, act, audio, set, remove }) {
  const [q, setQ] = useState(null)  // null = not searching
  const [sel, setSel] = useState(0)
  const hits = useMemo(() => (q ? search(C, q) : []), [q])
  const choose = i => { set(i); setQ(null) }
  const cls = act && act.r === r ? 'row' : act && act.c === r ? 'col' : ''
  return (
    <div className={`slot ${cls}`}>
      <span className="n">{r + 1}</span>
      <Play {...{ audio, i: idx, C }} />
      <div className="txt">
        <input type="text" aria-label={`Caption ${r + 1}, type to search`} value={q ?? C[idx].caption}
          onFocus={e => e.target.select()} onChange={e => { setQ(e.target.value); setSel(0) }} onBlur={() => setQ(null)}
          onKeyDown={e => {
            if (q === null) return
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); setSel((sel + (e.key === 'ArrowDown' ? 1 : hits.length - 1)) % Math.max(1, hits.length)) }
            else if (e.key === 'Enter' && hits.length) { e.preventDefault(); choose(hits[sel]); e.target.blur() }
            else if (e.key === 'Escape') e.target.blur()
          }} />
        <span className="g">#{idx + 1} · {genresOf(C[idx])}</span>
      </div>
      <div className="acts">
        <button title="Random caption" aria-label="Random caption" onClick={() => set(Math.floor(Math.random() * C.length))}>⟳</button>
        {n > 2 && <button title="Remove" aria-label="Remove caption" onClick={remove}>✕</button>}
      </div>
      {q && (
        <div className="pop">
          {hits.length ? hits.map((i, k) => (
            <div key={i} className={k === sel ? 'on' : ''} onMouseDown={e => { e.preventDefault(); choose(i) }}>
              <span className="g">#{i + 1} · {genresOf(C[i])}</span><br />{C[i].caption.slice(0, 170)}…
            </div>
          )) : <div className="g">No caption matches every word</div>}
        </div>
      )}
    </div>
  )
}

function Grid({ slots, view, F, sim, isNA, info, instr, pin, setPin, setHover }) {
  const n = slots.length, delta = view === 'delta'
  const fi = info(view === 'before' ? 'none' : instr)
  const value = (a, b) => (view === 'before' ? sim(F.f0, a, b) : view === 'after' ? sim(F.f1, a, b) : sim(F.f1, a, b) - sim(F.f0, a, b))
  const color = v => {
    if (delta) { const m = Math.max(Math.abs(fi.dlo), Math.abs(fi.dhi)), t = Math.max(-1, Math.min(1, v / m)); return [t >= 0 ? mix(WHITE, GREEN, t) : mix(WHITE, RED, -t), Math.abs(t)] }
    const t = Math.max(0, Math.min(1, (v - fi.lo) / (fi.hi - fi.lo))); return [mix(WHITE, RED, t), t]
  }
  const cells = []
  for (let c = 0; c < n; c++) cells.push(<div key={`h${c}`} className="h">{c + 1}</div>)
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const key = `${r}-${c}`
    if (c > r) { cells.push(<div key={key} />); continue }
    if (slots[r] === slots[c]) { cells.push(<div key={key} className="c d" />); continue }
    const pinned = pin && pin.r === r && pin.c === c
    const handlers = { 'data-r': r, 'data-c': c, onMouseEnter: () => setHover({ r, c }), onClick: () => setPin(pinned ? null : { r, c }) }
    if (isNA(slots[r]) || isNA(slots[c])) { cells.push(<div key={key} className={`c na ${pinned ? 'pin' : ''}`} {...handlers}>N/A</div>); continue }
    const v = value(slots[r], slots[c]), [col, t] = color(v)
    const txt = delta ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2).slice(1) : v.toFixed(2)
    cells.push(<div key={key} className={`c ${pinned ? 'pin' : ''}`} {...handlers} style={{ background: `rgb(${col})`, color: t > 0.55 ? '#fff' : '#111318' }}>{txt}</div>)
  }
  return <div className="mgrid" style={{ gridTemplateColumns: `repeat(${n}, var(--cell))` }} onMouseLeave={() => setHover(null)}>{cells}</div>
}

function Panels({ C, slots, act, pinned, unpin, F, sim, instr, facts, audio }) {
  const gC = useRef(null), gR = useRef(null)
  const place = () => {
    if (!act || !gC.current) return
    const el = document.querySelector(`.mgrid .c[data-r="${act.r}"][data-c="${act.c}"]`)
    if (!el) return
    const r = el.getBoundingClientRect(), vw = innerWidth, vh = innerHeight, pad = 12
    const cw = gC.current.offsetWidth, ch = gC.current.offsetHeight, rw = gR.current.offsetWidth, rh = gR.current.offsetHeight
    const cx = Math.min(Math.max(8, r.left), vw - cw - 8), cy = Math.max(8, r.top - ch - pad)
    let rx = r.right + pad, ry = Math.max(cy + ch + pad, Math.min(r.top, vh - rh - 8))
    if (rx + rw > vw - 8) rx = Math.max(8, r.left - rw - pad)
    if (ry + rh > vh - 8) ry = Math.max(8, vh - rh - 8)
    Object.assign(gC.current.style, { left: `${cx}px`, top: `${cy}px` })
    Object.assign(gR.current.style, { left: `${rx}px`, top: `${ry}px` })
  }
  useLayoutEffect(place)
  useEffect(() => {
    if (!pinned) return
    const onKey = e => e.key === 'Escape' && unpin()
    addEventListener('scroll', place, { passive: true }); addEventListener('keydown', onKey)
    return () => { removeEventListener('scroll', place); removeEventListener('keydown', onKey) }
  })
  if (!act || slots[act.r] === undefined || slots[act.c] === undefined) return null
  const a = slots[act.r], b = slots[act.c], v0 = sim(F.f0, a, b), v1 = sim(F.f1, a, b), d = v1 - v0
  const panel = (ref, i, chip, color, extra) => (
    <div ref={ref} className={`glass ${pinned ? 'pinned' : ''}`}>
      {pinned && <button className="close" aria-label="Unpin" onClick={unpin}>✕</button>}
      <div className="tag"><span className="chip" style={{ background: color }}>{chip}</span>{pinned && <Play {...{ audio, i, C }} />}<Meta C={C} i={i} /></div>
      {extra}
      <p>{C[i].caption}</p>
      {facts(i)}
    </div>
  )
  return (
    <>
      {panel(gC, b, act.c + 1, 'var(--ink)', <div className="vals"><span>{v0.toFixed(3)}</span>→<b>{v1.toFixed(3)}</b><span className={`delta ${d >= 0 ? 'up' : 'down'}`}>{d >= 0 ? '+' : '−'}{Math.abs(d).toFixed(3)}</span></div>)}
      {panel(gR, a, act.r + 1, 'var(--accent)')}
    </>
  )
}

function Lists({ C, M, K, N, F, slots, pin, instr, I, tab, setTab, topView, setTopView, sim, info, facts, audio }) {
  const r = Math.min(pin ? pin.r : 0, slots.length - 1), a = slots[r], i0 = info('none'), i1 = info(instr)
  const item = (rank, i, score, lohi, extra) => (
    <li key={i} className="item">
      <span className="rkc"><span className="rk">{rank}</span><Play {...{ audio, i, C }} /></span>
      <div className="cap"><p>{C[i].caption}</p><Meta C={C} i={i} />{facts(i)}</div>
      <div className="score"><b>{score.toFixed(3)}</b><span className="bar"><i style={{ width: `${Math.max(2, Math.min(100, (100 * (score - lohi.lo)) / (lohi.hi - lohi.lo)))}%` }} /></span>{extra}</div>
    </li>
  )
  const chg = (d, was, rank) => (
    <>
      <span className={`delta ${d >= 0 ? 'up' : 'down'}`}>{d >= 0 ? '+' : '−'}{Math.abs(d).toFixed(3)}</span>
      {was === rank ? <span className="moved">same rank</span> : <span className={`moved ${was > rank ? 'up' : 'down'}`}>{was > rank ? '▲' : '▼'} was #{was}</span>}
    </>
  )
  const source = (
    <div className="source">
      <span className="chip">{r + 1}</span><Play {...{ audio, i: a, C }} />
      <div><span className="label">Source caption{pin ? '' : ' · click a cell to choose its row'}</span><p>{C[a].caption}</p><Meta C={C} i={a} />{facts(a)}</div>
    </div>
  )
  let body
  if (tab === 'rank') {
    const pool = [...new Set(slots)].filter(b => b !== a)
    const o0 = [...pool].sort((x, y) => sim(F.f0, a, y) - sim(F.f0, a, x)), o1 = [...pool].sort((x, y) => sim(F.f1, a, y) - sim(F.f1, a, x))
    const p0 = new Map(o0.map((b, i) => [b, i + 1]))
    body = (
      <div className="panel">{source}
        <div className="two">
          <div><div className="colhead"><b>No instruction</b><span>General caption similarity</span></div><ol>{o0.map((b, i) => item(i + 1, b, sim(F.f0, a, b), i0))}</ol></div>
          <div><div className="colhead"><b>With instruction</b><span>{I[1]}</span></div><ol>{o1.map((b, i) => item(i + 1, b, sim(F.f1, a, b), i1, chg(sim(F.f1, a, b) - sim(F.f0, a, b), p0.get(b), i + 1)))}</ol></div>
        </div>
      </div>
    )
  } else if (tab === 'top') {
    let list = []
    if (topView === 'before') for (let k = 0; k < K; k++) { const b = F.f0.idx[a * K + k]; list.push(item(k + 1, b, deq(F.f0.sim[a * K + k], i0.lo, i0.hi), i0)) }
    else {
      const row0 = new Float32Array(N)
      for (let c = 0; c < N; c++) row0[c] = c === a ? -Infinity : sim(F.f0, a, c)
      for (let k = 0; k < K; k++) {
        const j = a * K + k, b = F.f1.idx[j], s1 = deq(F.f1.sim[j], i1.lo, i1.hi), s0 = row0[b]
        let was = 1
        for (let c = 0; c < N; c++) was += row0[c] > s0
        list.push(item(k + 1, b, s1, i1, chg(s1 - s0, was, k + 1)))
      }
    }
    body = (
      <div className="panel">
        <div className="row-between">
          <Seg label="Top 10 view" value={topView} onChange={setTopView} options={[['before', 'No instruction'], ['after', 'With instruction']]} />
          <span className="hint">{topView === 'after' ? 'Exact neighbours with the instruction; change and old rank against no instruction' : 'Exact neighbours without an instruction'}</span>
        </div>
        {source}<ol>{list}</ol>
      </div>
    )
  } else body = <div className="panel"><About M={M} N={N} /></div>
  return (
    <section className="card lists">
      <div className="tabs" role="tablist">
        {[['rank', 'Ranking of these captions'], ['top', 'Top 10 in the dataset'], ['about', 'About the data']].map(([t, l]) => (
          <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>{l}</button>
        ))}
      </div>
      {body}
    </section>
  )
}

function About({ M, N }) {
  const errs = Object.values(M.files).map(f => f.cell_abs_err).filter(x => x != null), err = errs.reduce((a, b) => a + b, 0) / errs.length
  return (
    <div className="about">
      <p><b>Data.</b> {N.toLocaleString()} Music4All tracks with ACE-Step captions, at most 30 tracks per first genre. Every teacher embeds each caption together with the instruction and compares captions under the same instruction.</p>
      <p><b>Teachers.</b> Qwen3-Embedding-8B with the instruction worded as a question; InBedder Llama-2-7B, whose embedding is the hidden state that would produce the answer; the same InBedder told it may answer “none”; and an LLM extraction teacher, where Qwen3.5-9B writes the answer and Qwen3-Embedding-8B embeds it.</p>
      <p><b>Not applicable.</b> JEV-9B gives a calibrated probability that the caption answers the question. Embedder confidence is a caption's similarity to the null caption “A piece of music.” under the same instruction. AUROC in the N/A panel measures how well that confidence finds the captions JEV or the LLM call unanswerable.</p>
      <p><b>Precision.</b> Matrix cells and the selected-caption ranking use {M.dim}-dimension compressed embeddings (mean absolute error {err.toFixed(4)} in cosine). The top 10 lists, their scores and the colour ranges are exact; the change and old rank in the top 10 use the compressed no-instruction embeddings.</p>
      <p><b>Audio.</b> The 30-second Music4All previews the captions describe stream from the project's audio Space; this site stores no audio. Spotify links open the full track.</p>
    </div>
  )
}

function NowPlaying({ audio, C }) {
  const i = audio.idx, c = C[i]
  return (
    <div className="np" role="region" aria-label="Now playing">
      <button className="play" aria-label={audio.playing ? 'Pause' : 'Play'} onClick={() => audio.toggle(i, c.id)}>{audio.loading ? <span className="spin dark" /> : audio.playing ? '❚❚' : '▶'}</button>
      <div className="t">
        <span><b>#{i + 1}</b> · {genresOf(c)} — {audio.error ? 'Could not load this clip' : c.caption}</span>
        <span className="prog" onClick={e => { const r = e.currentTarget.getBoundingClientRect(); audio.seek((e.clientX - r.left) / r.width) }}>
          <i style={{ width: `${(100 * audio.t) / (audio.dur || 30)}%` }} />
        </span>
      </div>
      {LIVE_WAVE ? <Wave audio={audio} /> : <span className="num npt">{Math.floor(audio.t)}s</span>}
      <button className="x" aria-label="Stop" onClick={audio.stop}>✕</button>
    </div>
  )
}
