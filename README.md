# Conditional Similarity Probe

Can instruction-conditioned text embedders act as teachers for instruction-conditioned music similarity?
2,000 Music4All captions (ACE-Step captioner) are embedded under 30 similarity instructions by four teachers
(Qwen3-Embedding-8B with question wording, InBedder Llama-2-7B, InBedder allowed to answer "none", and LLM extraction
with Qwen3.5-9B), with JEV-9B estimating whether each caption can answer each instruction at all.

- `probe_teachers.py`, `probe.sbatch`: the pipeline (sample, embed, gen, jev, llm, export), run on SLURM.
- `website/`: the interactive viewer (React + Vite), deployed to GitHub Pages by `.github/workflows/deploy-website.yml`.
  `website/public/data/` is the `export` stage output. Audio streams from the Music Lenses Hugging Face Space; this
  repository stores no audio.
- `viewer.html`: the single-file version published as a Claude artifact.

```bash
cd website && npm ci && npm run dev        # local (audio plays; the live waveform needs the pliploop.github.io origin)
npm run build && npm run ssr-check         # what CI runs
```
