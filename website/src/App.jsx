import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { AUDIO, GREEN, MODELS, NA_MODES, SHORT, NOT_MENTIONED, RED, WHITE, cell, deq, getFile, getJev, getJson, kindOf, mix, search, trimLoop } from './lib.js'

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
      ctx.lineWidth = 3; ctx.strokeStyle = 'rgb(196,36,48)'; ctx.lineJoin = 'round'
      ctx.beginPath()
      for (let x = 0; x < w; x++) {
        const v = (buf[Math.floor((x / w) * buf.length)] - 128) / 128
        const y = h / 2 + v * h * 0.46
        x ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
      }
      ctx.stroke()
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [audio.playing, audio.idx])
  return <canvas ref={cv} width={280} height={72} className="wave" aria-hidden="true" />
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
const PlayIcon = () => <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3.2v9.6a.6.6 0 0 0 .9.5l7.6-4.8a.6.6 0 0 0 0-1L5.9 2.7a.6.6 0 0 0-.9.5Z" fill="currentColor" /></svg>
const PauseIcon = () => <svg viewBox="0 0 16 16" aria-hidden="true"><rect x="4" y="3" width="2.8" height="10" rx="1" fill="currentColor" /><rect x="9.2" y="3" width="2.8" height="10" rx="1" fill="currentColor" /></svg>
function Play({ audio, i, C }) {
  const on = audio.idx === i && (audio.playing || audio.loading)
  return (
    <button className={`play ${on ? 'on' : ''}`} aria-label={`${on ? 'Pause' : 'Play'} track #${i + 1}`}
      onClick={e => { e.stopPropagation(); audio.toggle(i, C[i].id) }}>
      {audio.idx === i && audio.loading ? <span className="spin" /> : on ? <PauseIcon /> : <PlayIcon />}
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
  const [side, setSide] = useState(() => typeof window === 'undefined' || window.innerWidth > 900)
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
  useEffect(() => {
    if (!M) return
    const order = M.instructions.map(x => x[0]).filter(x => x !== 'none')
    const onKey = e => {
      if (e.target.closest?.('input, textarea, select') || (e.key !== '[' && e.key !== ']')) return
      setInstr(cur => order[(order.indexOf(cur) + (e.key === ']' ? 1 : -1) + order.length) % order.length])
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [M])
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

  if (err) return <div className="boot"><p>Could not load data: {err}</p></div>
  if (!D) return <div className="boot"><span className="spin dark" /><p>Loading the probe…</p></div>

  const I = M.instructions.find(x => x[0] === instr)
  const info = key => M.files[`${model}/${key}`]
  const sim = (f, a, b) => cell(f, DIM, a, b)

  function facts(i) {
    if (!ready) return null
    const p = jevP(i), a = appSim(i), g = model.startsWith('inbedder') ? F.gens[model] : null
    return (
      <dl className="facts">
        {isNA(i) && <div className="na-line">Not applicable to this instruction</div>}
        {g && <div><dt>InBedder</dt><dd>{trimLoop(g[i])}</dd></div>}
        {F.llm && <div><dt>LLM</dt><dd>{F.llm[i]}</dd></div>}
        {p !== null && <div><dt>P(answerable)</dt><dd className="num">{p.toFixed(2)}</dd></div>}
        {a !== null && <div><dt>Null-caption sim</dt><dd className="num">{a.toFixed(2)}</dd></div>}
      </dl>
    )
  }

  return (
    <div className={`app ${side ? 'open' : 'closed'}`}>
      <Sidebar {...{ M, model, setModel, instr, setInstr, na, setNa, avail, naCount, thr, setThr, fi: info(instr), hasApp: !!F?.f1?.app && ready, close: () => setSide(false) }} />
      <div className="scrim" onClick={() => setSide(false)} />
      <main className="main">
        <div className="topbar">
          <button className="icon" aria-label={side ? 'Hide controls' : 'Show controls'} aria-expanded={side} onClick={() => setSide(!side)}>
            <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><rect x="2" y="3" width="14" height="12" rx="3" fill="none" stroke="currentColor" strokeWidth="1.5" /><path d="M7 3v12" stroke="currentColor" strokeWidth="1.5" /></svg>
          </button>
          <span className="crumb">{MODELS[model]?.[0] || model}</span>
          <span className="crumb muted">{ready ? `${N.toLocaleString()} captions` : 'Loading…'}</span>
        </div>

        <section className="headline" aria-live="polite">
          <p className="eyebrow">{kindOf(I[3])} · {I[2]}</p>
          <InstrPicker {...{ M, instr, setInstr }} />
        </section>

        <section className="stage" aria-label="Similarity matrix">
          <div className="stage-head">
            <Seg label="Matrix view" value={view} onChange={setView} options={[['before', 'Plain'], ['after', 'Instructed'], ['delta', 'Change']]} />
            {ready && <Legend view={view} fi={info(view === 'before' ? 'none' : instr)} />}
          </div>
          <div className="scroller">
            <div className="board">
              <Slots {...{ C, slots, setSlots, act: pin || hover, audio, setPin }} />
              {ready ? <Grid {...{ slots, view, F, sim, isNA, info, instr, pin, setPin, setHover, act: pin || hover }} /> : <div className="grid-wait"><span className="spin dark" /></div>}
            </div>
          </div>
          <div className="slot-actions">
            <button className="text" onClick={() => setSlots([...slots, Math.floor(Math.random() * N)])}>Add caption</button>
            <button className="text" onClick={() => { setSlots(slots.map(() => Math.floor(Math.random() * N))); setPin(null) }}>Shuffle</button>
          </div>
        </section>

        {ready && <Lists {...{ C, M, K, N, F, slots, pin, instr, I, tab, setTab, topView, setTopView, sim, info, facts, audio }} />}
      </main>

      {ready && <Panels {...{ C, slots, act: pin || hover, pinned: !!pin, unpin: () => setPin(null), F, sim, instr, facts, audio }} />}
      {audio.idx !== null && <NowPlaying {...{ audio, C }} />}
    </div>
  )
}

function InstrPicker({ M, instr, setInstr }) {
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState('')
  const ref = useRef(null)
  const I = M.instructions.find(x => x[0] === instr)
  useEffect(() => {
    if (!open) return
    const close = e => { if (!ref.current?.contains(e.target)) setOpen(false) }
    const esc = e => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', close); addEventListener('keydown', esc)
    return () => { document.removeEventListener('mousedown', close); removeEventListener('keydown', esc) }
  }, [open])
  const match = x => !q || `${x[1]} ${x[2]} ${SHORT[x[0]] || ''}`.toLowerCase().includes(q.toLowerCase())
  const groups = M.instructions.filter(x => x[0] !== 'none' && match(x)).reduce((g, x) => ((g[kindOf(x[3])] ||= []).push(x), g), {})
  const pick = iid => { setInstr(iid); setOpen(false); setQ('') }
  return (
    <div className="picker" ref={ref}>
      <button className="picker-btn" aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen(!open)}>
        <h1 key={instr}>{I[1]}</h1>
        <svg className="chev" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </button>
      {open && (
        <div className="menu" role="listbox" aria-label="Instructions">
          <input autoFocus placeholder="Filter instructions" value={q} onChange={e => setQ(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') { const first = Object.values(groups)[0]?.[0]; if (first) pick(first[0]) } }} />
          <div className="menu-list">
            {Object.entries(groups).map(([g, xs]) => (
              <div key={g}>
                <h4>{g}</h4>
                {xs.map(x => (
                  <button key={x[0]} role="option" aria-selected={x[0] === instr} onClick={() => pick(x[0])}>
                    <span>{x[1]}</span><small>{x[2]}</small>
                  </button>
                ))}
              </div>
            ))}
            {!Object.keys(groups).length && <p className="meta">No instruction matches.</p>}
          </div>
        </div>
      )}
    </div>
  )
}

function Legend({ view, fi }) {
  if (!fi) return null
  const delta = view === 'delta', m = delta ? Math.max(Math.abs(fi.dlo), Math.abs(fi.dhi)) : 0
  return (
    <div className="legend" title={delta ? 'Change in cosine from adding the instruction' : 'Cosine, min to max over the whole dataset'}>
      <span className="num">{delta ? `−${m.toFixed(2)}` : fi.lo.toFixed(2)}</span>
      <i style={{ background: delta ? `linear-gradient(90deg,rgb(${RED}),#fff,rgb(${GREEN}))` : `linear-gradient(90deg,#fff,rgb(${RED}))` }} />
      <span className="num">{delta ? `+${m.toFixed(2)}` : fi.hi.toFixed(2)}</span>
    </div>
  )
}

function Sidebar({ M, model, setModel, instr, setInstr, na, setNa, avail, naCount, thr, setThr, fi, hasApp, close }) {
  const groups = M.instructions.filter(x => x[0] !== 'none').reduce((g, x) => ((g[kindOf(x[3])] ||= []).push(x), g), {})
  const fmt = x => (x == null ? '–' : x.toFixed(2))
  return (
    <aside className="side" aria-label="Controls">
      <div className="brand">
        <span className="mark" aria-hidden="true" />
        <span>Similarity probe</span>
        <button className="icon only-mobile" aria-label="Close controls" onClick={close}>✕</button>
      </div>

      <div className="sec">
        <h3>Teacher</h3>
        {M.models.map(k => (
          <button key={k} className="row" aria-pressed={k === model} onClick={() => setModel(k)}>
            <span>{MODELS[k]?.[0] || k}</span><small>{MODELS[k]?.[1]}</small>
          </button>
        ))}
      </div>

      <div className="sec">
        <h3>Instruction <kbd>[</kbd><kbd>]</kbd></h3>
        {Object.entries(groups).map(([g, xs]) => (
          <div key={g} className="group">
            <h4>{g}</h4>
            {xs.map(([iid, s]) => (
              <button key={iid} className="row" aria-pressed={iid === instr} title={s} onClick={() => setInstr(iid)}><span>{SHORT[iid] || s}</span></button>
            ))}
          </div>
        ))}
      </div>

      <div className="sec">
        <h3>Not applicable {na !== 'off' && <em>{naCount.toLocaleString()}</em>}</h3>
        {NA_MODES.map(([v, t, s]) => {
          const off = v !== 'off' && !avail[v]
          return (
            <button key={v} className="row" aria-pressed={na === v} disabled={off} title={off ? 'Not computed for this data' : s} onClick={() => setNa(v)}>
              <span>{t}</span>
            </button>
          )
        })}
        {na === 'anchor' && hasApp && thr !== null && (
          <label className="slider">
            <span>Threshold <b className="num">{thr.toFixed(3)}</b></span>
            <input type="range" min={fi.anchor_lo} max={fi.anchor_hi} step="0.001" value={thr} onChange={e => setThr(+e.target.value)} />
          </label>
        )}
        <dl className="mini">
          <div><dt>Confidence vs JEV</dt><dd className="num">{fmt(fi?.auroc_anchor_vs_jev)}</dd></div>
          <div><dt>Confidence vs LLM</dt><dd className="num">{fmt(fi?.auroc_anchor_vs_llm)}</dd></div>
          <div><dt>JEV vs LLM</dt><dd className="num">{fmt(fi?.auroc_jev_vs_llm)}</dd></div>
        </dl>
        <p className="foot">AUROC of N/A detection for this teacher and instruction.</p>
      </div>
    </aside>
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
  const cls = act && act.r === r ? 'row-on' : act && act.c === r ? 'col-on' : ''
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

function Grid({ slots, view, F, sim, isNA, info, instr, pin, setPin, setHover, act }) {
  const n = slots.length, delta = view === 'delta'
  const fi = info(view === 'before' ? 'none' : instr)
  const value = (a, b) => (view === 'before' ? sim(F.f0, a, b) : view === 'after' ? sim(F.f1, a, b) : sim(F.f1, a, b) - sim(F.f0, a, b))
  const color = v => {
    if (delta) { const m = Math.max(Math.abs(fi.dlo), Math.abs(fi.dhi)), t = Math.max(-1, Math.min(1, v / m)); return [t >= 0 ? mix(WHITE, GREEN, t) : mix(WHITE, RED, -t), Math.abs(t)] }
    const t = Math.max(0, Math.min(1, (v - fi.lo) / (fi.hi - fi.lo))); return [mix(WHITE, RED, t), t]
  }
  const cells = []
  for (let c = 0; c < n; c++) cells.push(<div key={`h${c}`} className={`h ${act && act.c === c ? 'on' : ''}`}>{c + 1}</div>)
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
      <div><span className="label">{pin ? `Row ${r + 1}` : `Row ${r + 1} · click a cell to change`}</span><p>{C[a].caption}</p><Meta C={C} i={a} />{facts(a)}</div>
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
          <div><div className="colhead"><b>Plain</b><span>no instruction</span></div><ol>{o0.map((b, i) => item(i + 1, b, sim(F.f0, a, b), i0))}</ol></div>
          <div><div className="colhead"><b>Instructed</b><span>{SHORT[instr] || I[1]}</span></div><ol>{o1.map((b, i) => item(i + 1, b, sim(F.f1, a, b), i1, chg(sim(F.f1, a, b) - sim(F.f0, a, b), p0.get(b), i + 1)))}</ol></div>
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
          <Seg label="Top 10 view" value={topView} onChange={setTopView} options={[['before', 'Plain'], ['after', 'Instructed']]} />
        </div>
        {source}<ol>{list}</ol>
      </div>
    )
  } else body = <div className="panel"><About M={M} N={N} /></div>
  return (
    <section className="lists">
      <div className="tabs" role="tablist">
        {[['rank', 'Ranking'], ['top', 'Nearest in dataset'], ['about', 'About']].map(([t, l]) => (
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

const fmtT = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`
function NowPlaying({ audio, C }) {
  const i = audio.idx, c = C[i]
  return (
    <div className="np" role="region" aria-label="Now playing">
      <button className="np-play" aria-label={audio.playing ? 'Pause' : 'Play'} onClick={() => audio.toggle(i, c.id)}>
        {audio.loading ? <span className="spin" /> : audio.playing ? <PauseIcon /> : <PlayIcon />}
      </button>
      <div className="t">
        <span className="t1"><b>#{i + 1}</b><span>{genresOf(c)}</span></span>
        <span className="t2">{audio.error ? 'Could not load this clip' : c.caption}</span>
        <span className="tl">
          <span className="num">{fmtT(audio.t)}</span>
          <span className="prog" onClick={e => { const r = e.currentTarget.getBoundingClientRect(); audio.seek((e.clientX - r.left) / r.width) }}>
            <i style={{ width: `${(100 * audio.t) / (audio.dur || 30)}%` }} />
          </span>
          <span className="num">{fmtT(audio.dur || 30)}</span>
        </span>
      </div>
      {LIVE_WAVE && <Wave audio={audio} />}
      <button className="x" aria-label="Stop" onClick={audio.stop}>✕</button>
    </div>
  )
}
