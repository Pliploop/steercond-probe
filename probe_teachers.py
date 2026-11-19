"""Are instruction-conditioned text embedders usable as teachers for instruction-conditioned music similarity?

2k Music4All ACE-Step captions x 30 similarity instructions. Teachers embed every caption with the instruction,
sim(T(c_i, u), T(c_j, u)). Applicability ("does this caption say anything about u?") comes from three places:
JEV-9B's calibrated P(answerable), each embedder's similarity to a null-caption anchor, and InBedder's literal "none".

Stages (GPU stages shard over instructions: --shard / SLURM_ARRAY_TASK_ID, --nshards):
    sample                      login node: pick captions -> OUT/captions.json
    embed --model M             caption + null-anchor embeddings -> OUT/emb/M/<iid>.npy, <iid>.anchor.npy
    gen --model M               InBedder short answers (vLLM) -> OUT/gen/M/<iid>.json
    jev                         JEV-9B P(answerable) per caption -> OUT/jev/<iid>.npy
    llm                         Qwen3.6-27B extracted answers -> OUT/llm/<iid>.json (feeds the llm_extract teacher)
    export                      GPU: dataset-wide stats, exact top-10s, compressed embeddings -> OUT/web/
"""
import argparse
import base64
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np

OUT = os.environ.get("PROBE_OUT", "/gpfs/scratch/acw749/steercond/probe2k")
CAPS = "/gpfs/scratch/acw749/nclap/captions/acestep/shard_{:03d}.jsonl"
M4A = "/data/EECS-Pauwels-C4DM/music4all/"
N = 2000
GENRE_CAP = 30  # at most this many tracks per first genre, for diversity
K = 10  # exact neighbours stored per caption
DIM = 96  # compressed embedding size for the in-browser matrix

# (id, similarity phrasing, question phrasing, kind)
INSTRUCTIONS = [
    ("genre", "Find tracks of a similar genre.", "What is the genre of this track?", "attribute"),
    ("style_para", "Find tracks in the same musical style.", "What musical style is this?", "paraphrase:genre"),
    ("instrumentation", "Find tracks with similar instrumentation.", "What instruments are playing?", "attribute"),
    ("instr_para", "Find tracks that use the same instruments.", "Which instruments can be heard?", "paraphrase:instrumentation"),
    ("mood", "Find tracks with a similar mood or emotion.", "What is the mood of this track?", "attribute"),
    ("energy", "Find tracks with a similar tempo and energy level.", "How fast and energetic is this track?", "attribute"),
    ("vocal_style", "Find tracks with a similar vocal delivery style.", "How are the vocals delivered?", "attribute"),
    ("voice", "Find tracks whose singer has a similar voice (gender, timbre, register).",
     "What does the lead singer's voice sound like?", "attribute"),
    ("rhythm", "Find tracks with a similar rhythm and groove.", "What is the rhythm or groove like?", "attribute"),
    ("production", "Find tracks with similar production and sound texture, e.g. lo-fi versus polished.",
     "What is the production quality and sound texture?", "attribute"),
    ("era", "Find tracks that sound like they come from the same era or decade.",
     "What era or decade does this track sound like?", "attribute"),
    ("culture", "Find tracks from a similar cultural or geographic musical tradition.",
     "What cultural or geographic tradition is this music from?", "attribute"),
    ("lead_instr", "Find tracks with the same lead melodic instrument.", "What is the lead melodic instrument?", "attribute"),
    ("drums", "Find tracks with similar drums and percussion.", "What do the drums and percussion sound like?", "attribute"),
    ("bass", "Find tracks with a similar bass sound.", "What does the bass sound like?", "attribute"),
    ("guitar_tone", "Find tracks with a similar guitar tone.", "What does the guitar sound like?", "attribute"),
    ("harmony", "Find tracks with a similar harmonic character (major or minor, consonant or dissonant).",
     "Is the harmony major or minor, consonant or dissonant?", "attribute"),
    ("use_case", "Find tracks that would fit the same use case, such as the same film scene, playlist or setting.",
     "What setting, scene or playlist would this track fit?", "attribute"),
    ("intensity", "Find tracks with a similar level of aggression and intensity.", "How aggressive or intense is this track?", "attribute"),
    ("structure", "Find tracks with a similar song structure and dynamic arc.", "How does the track's structure and dynamics evolve?", "attribute"),
    ("q_vocals", "Group tracks by whether they are instrumental or have vocals.", "Is this track instrumental or does it have vocals?", "question"),
    ("q_guitar_solo", "Group tracks by whether they feature a guitar solo.", "Does this track feature a guitar solo?", "question"),
    ("q_brass", "Group tracks by whether they feature saxophone or brass.", "Is there a saxophone or brass in this track?", "question"),
    ("q_acoustic", "Group tracks by whether they are acoustic or electronic.", "Is this track acoustic or electronic?", "question"),
    ("q_gender", "Group tracks by whether the lead vocalist is male or female.", "Is the lead vocalist male or female?", "question"),
    ("q_dance", "Group tracks by whether they are danceable.", "Is this track danceable?", "question"),
    ("q_synth", "Group tracks by whether they use synthesizers.", "Does this track use synthesizers?", "question"),
    ("q_piano", "Group tracks by whether they feature piano or keyboards.", "Is there a piano or keyboard in this track?", "question"),
    ("q_happy", "Group tracks by whether they sound happy or sad.", "Does this track sound happy or sad?", "question"),
    ("q_language", "Group tracks by the language of the vocals.", "What language are the vocals in?", "question"),
]
ALL = [("none", None, None, "baseline")] + INSTRUCTIONS
Q = {iid: q for iid, _, q, _ in INSTRUCTIONS}

MODELS = {
    "qwen3emb8b_q": "Qwen/Qwen3-Embedding-8B",
    "inbedder7b": "BrandonZYW/llama-2-7b-InBedder",
    "inbedder7b_na": "BrandonZYW/llama-2-7b-InBedder",  # same model, allowed to answer "none"
    "llm_extract": "Qwen/Qwen3-Embedding-8B",  # embeds the LLM stage's answers
}
LLM = "Qwen/Qwen3.5-9B"  # fits a 40 GB A100 (Qwen3.6-27B-FP8 needed 80 GB)
JEV = "autotrust/JEV-9B"
INBEDDER_PATTERN = "### Input:\n{input}\n\n### Instruction:\n{instruction}\n\n### Response:"
INBEDDER_NONE = "What does this text describe?"
NA_SUFFIX = " If the description does not say, answer none."
NULL_CAPTION = "A piece of music."
CLIP_AT, CLIP_S, PER_SPRITE = 10, 10, 50  # page audio: seconds 10-20 of each 30 s preview, 50 clips per mp3
# the LLM's "no information" answers; Qwen3.5-9B often paraphrases the requested "not mentioned"
NOT_MENTIONED = __import__("re").compile(r"^not mentioned|\b(does not|doesn't|do not) (state|specify|mention|imply|indicate|say)|\bnot (specified|stated|mentioned|indicated)", 2)  # anchor: a caption that answers no question


def local(repo):
    """Cached snapshot path. vLLM's snapshot_download rejects caches missing files we skipped (optimizer states, vision)."""
    import glob
    return glob.glob(f"{os.environ['HF_HOME']}/hub/models--{repo.replace('/', '--')}/snapshots/*")[0]


def read_tsv(name):
    return {r["id"]: r for r in csv.DictReader(open(M4A + name), delimiter="\t")}


def sample():
    caps = {}
    for s in range(16):
        for line in open(CAPS.format(s)):
            d = json.loads(line)
            if d["source"] == "music4all" and d["caption"]:
                caps[d["id"]] = d["caption"]
    genres, tags, meta, lang = (read_tsv(f) for f in ("id_genres.csv", "id_tags.csv", "id_metadata.csv", "id_lang.csv"))
    ids = sorted(caps)
    random.Random(0).shuffle(ids)
    per, picked = defaultdict(int), []
    for i in ids:
        g = genres.get(i, {}).get("genres", "").split(",")[0]
        if g and per[g] < GENRE_CAP and i in meta:
            per[g] += 1
            picked.append(i)
        if len(picked) == N:
            break
    rows = []
    for i in picked:
        m = meta[i]
        rows.append({"id": i, "caption": caps[i], "genres": genres[i]["genres"].split(",")[:4],
                     "tags": tags.get(i, {}).get("tags", "").split(",")[:6], "lang": lang.get(i, {}).get("lang", ""),
                     **{k: round(float(m[k]), 3) for k in ("danceability", "energy", "valence", "tempo", "mode", "release")}})
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(f"{OUT}/captions.json", "w"))
    print(len(rows), "captions,", len(per), "first-genres")


def load():
    return json.load(open(f"{OUT}/captions.json"))


def my_iids(args, skip_none=False):
    ids = [x[0] for x in ALL if not (skip_none and x[0] == "none")]
    return ids[args.shard::args.nshards]


def texts_for(model, iid, caps, answers=None):
    """Teacher inputs for captions `caps` under instruction iid."""
    q = Q.get(iid)
    if model == "qwen3emb8b_q":
        return [c if q is None else f"Instruct: Identify the answer to this question about the given music description: {q}\nQuery:{c}"
                for c in caps]
    if model.startswith("inbedder7b"):
        ins = INBEDDER_NONE if q is None else q + (NA_SUFFIX if model.endswith("_na") else "")
        return [INBEDDER_PATTERN.format(input=c, instruction=ins) for c in caps]
    if model == "llm_extract":  # answers aligned with caps; the question prefix keeps "yes"/"no" on topic
        return [c if q is None else f"{q} {a}" for c, a in zip(caps, answers)]
    raise ValueError(model)


def embed(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    name, rows = args.model, load()
    causal = name.startswith("inbedder")
    tok = AutoTokenizer.from_pretrained(MODELS[name], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    model = (AutoModelForCausalLM if causal else AutoModel).from_pretrained(MODELS[name], dtype=torch.bfloat16).cuda().eval()
    os.makedirs(f"{OUT}/emb/{name}", exist_ok=True)

    @torch.no_grad()
    def enc(texts, bs=64):
        order = np.argsort([len(t) for t in texts])  # length-sorted batches waste less padding
        out = None
        for b in range(0, len(texts), bs):
            idx = order[b:b + bs]
            x = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            h = (model(**x, output_hidden_states=True).hidden_states[-1] if causal else model(**x).last_hidden_state)[:, -1]
            if out is None:
                out = np.zeros((len(texts), h.shape[-1]), np.float16)
            out[idx] = F.normalize(h.float(), dim=-1).cpu().numpy()
        return out

    caps = [r["caption"] for r in rows]
    for iid in my_iids(args):
        f = f"{OUT}/emb/{name}/{iid}.npy"
        if os.path.exists(f):
            continue
        ans = json.load(open(f"{OUT}/llm/{iid}.json")) if name == "llm_extract" and iid != "none" else [None] * len(caps)
        np.save(f, enc(texts_for(name, iid, caps, ans)))
        if iid != "none":
            null_ans = ["not mentioned"] if name == "llm_extract" else None
            np.save(f"{OUT}/emb/{name}/{iid}.anchor.npy", enc(texts_for(name, iid, [NULL_CAPTION], null_ans)))
        print(name, iid, flush=True)


def gen(args):
    from vllm import LLM as V, SamplingParams
    name, rows = args.model, load()
    caps = [r["caption"] for r in rows]
    os.makedirs(f"{OUT}/gen/{name}", exist_ok=True)
    iids = [i for i in my_iids(args, skip_none=True) if not os.path.exists(f"{OUT}/gen/{name}/{i}.json")]
    if not iids:
        return
    llm = V(model=local(MODELS[name]), max_model_len=1024, gpu_memory_utilization=0.9, seed=0)
    prompts = [t for iid in iids for t in texts_for(name, iid, caps)]
    res = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=48, stop=["\n"]))
    for k, iid in enumerate(iids):
        json.dump([o.outputs[0].text.strip() for o in res[k * len(caps):(k + 1) * len(caps)]], open(f"{OUT}/gen/{name}/{iid}.json", "w"))


def jev(args):
    """JEV-9B System 1 (noul): P(true) that the caption answers the question. LoRA merged by hand (no peft in env)."""
    import torch
    import glob
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, root = load(), local(JEV)
    os.makedirs(f"{OUT}/jev", exist_ok=True)
    iids = [i for i in my_iids(args, skip_none=True) if not os.path.exists(f"{OUT}/jev/{i}.npy")]
    if not iids:
        return
    tok = AutoTokenizer.from_pretrained(root)
    model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.bfloat16).cuda().eval()
    cfg = json.load(open(f"{root}/adapter/adapter_config.json"))
    lora, params = load_file(f"{root}/adapter/adapter_model.safetensors"), dict(model.named_parameters())
    for k, A in lora.items():
        if ".lora_A." in k:
            B = lora[k.replace(".lora_A.", ".lora_B.")]
            p = params[k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")]
            p.data += (B.float() @ A.float() * (cfg["lora_alpha"] / cfg["r"])).to(p.dtype).to(p.device)
    head = load_file(f"{root}/head.safetensors")
    W, b = head["proj.weight"].cuda(), head["proj.bias"].cuda()
    T = json.load(open(f"{root}/calibration.json"))["per_kind"]["noul"]
    for iid in iids:
        prompts = [f"[kind] noul\n[state] {r['caption']}\n[question] Does this music description state or clearly imply the "
                   f"answer to the question \"{Q[iid]}\"?\n[options]\nfalse\ntrue\n[decision]:" for r in rows]
        ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        # Qwen3.5 mixes linear-attention layers, so batch only prompts of identical length (no padding at all)
        by_len = defaultdict(list)
        for i, x in enumerate(ids):
            by_len[len(x)].append(i)
        p_true = np.zeros(len(rows), np.float32)
        with torch.no_grad():
            for L, idx in by_len.items():
                for s in range(0, len(idx), 48):
                    sub = idx[s:s + 48]
                    h = model.model(input_ids=torch.tensor([ids[i] for i in sub], device="cuda")).last_hidden_state[:, -1].float()
                    z = (h @ W.T + b) / T
                    p_true[sub] = torch.softmax(z[:, 0:2], -1)[:, 1].cpu().numpy()
        np.save(f"{OUT}/jev/{iid}.npy", p_true)
        print("jev", iid, f"mean P(answerable)={p_true.mean():.3f}", flush=True)


def llm(args):
    from vllm import LLM as V, SamplingParams
    rows = load()
    os.makedirs(f"{OUT}/llm", exist_ok=True)
    iids = [i for i in my_iids(args, skip_none=True) if not os.path.exists(f"{OUT}/llm/{i}.json")]
    if not iids:
        return
    llm = V(model=local(LLM), max_model_len=2048, gpu_memory_utilization=0.9, seed=0, enable_prefix_caching=True,
            limit_mm_per_prompt={"image": 0, "video": 0})  # text only
    convs = []
    for r in rows:  # caption-major order so the caption prefix is reused across this shard's questions
        for iid in iids:
            convs.append([{"role": "user", "content":
                f"Music description:\n{r['caption']}\n\nQuestion: {Q[iid]}\n\nAnswer in a short phrase (at most 15 words), using "
                "only what the description states or clearly implies. If the description gives no information to answer, "
                "reply exactly: not mentioned"}])
    res = llm.chat(convs, SamplingParams(temperature=0, max_tokens=40), chat_template_kwargs={"enable_thinking": False})
    out = {iid: [] for iid in iids}
    for k, o in enumerate(res):
        out[iids[k % len(iids)]].append(o.outputs[0].text.strip())
    for iid, a in out.items():
        json.dump(a, open(f"{OUT}/llm/{iid}.json", "w"))


def audio(args):
    """Playable clips for the page: artifacts serve no external media, so 10 s mono clips are packed 50 per CBR mp3 sprite
    (clip k of sprite s starts at k * CLIP_S); also adds each track's Spotify id for a full-track link."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    ff, sr, rows, W = "/data/home/acw749/conda-envs/steer/bin/ffmpeg", 22050, load(), f"{OUT}/web"
    os.makedirs(f"{W}/audio", exist_ok=True)

    def clip(r):  # seconds [CLIP_AT, CLIP_AT + CLIP_S) of the 30 s preview, zero-padded to exact length
        pcm = subprocess.run([ff, "-v", "error", "-ss", str(CLIP_AT), "-t", str(CLIP_S), "-i", f"{M4A}audios/{r['id']}.mp3",
                              "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout
        x = np.frombuffer(pcm, np.float32)[:CLIP_S * sr]
        return np.pad(x, (0, CLIP_S * sr - len(x)))

    with ThreadPoolExecutor(8) as ex:
        clips = list(ex.map(clip, rows))
    for s in range(0, len(rows), PER_SPRITE):
        pcm = np.concatenate(clips[s:s + PER_SPRITE]).astype(np.float32).tobytes()
        subprocess.run([ff, "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-c:a", "libmp3lame", "-b:a", "48k",
                        f"{W}/audio/s{s // PER_SPRITE:03d}.mp3"], input=pcm, check=True)
    spotify = {i: r["spotify_id"] for i, r in read_tsv("id_metadata.csv").items()}
    for r in rows:
        r["sp"] = spotify.get(r["id"], "")
    json.dump(rows, open(f"{W}/captions.json", "w"))
    meta = json.load(open(f"{W}/meta.json"))
    meta["audio"] = {"clip_s": CLIP_S, "clip_at": CLIP_AT, "per": PER_SPRITE, "n_sprites": -(-len(rows) // PER_SPRITE)}
    json.dump(meta, open(f"{W}/meta.json", "w"))
    print("audio sprites", meta["audio"], "silent clips", sum(float(np.abs(c).max()) < 1e-4 for c in clips))


def export(args):
    import torch
    from sklearn.metrics import roc_auc_score

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load()
    n, W = len(rows), f"{OUT}/web"
    for d in ("bin", "gen", "llm"):
        os.makedirs(f"{W}/{d}", exist_ok=True)
    meta_f = f"{W}/meta.json"
    meta = json.load(open(meta_f)) if os.path.exists(meta_f) and not args.fresh else {"files": {}}
    meta.update({"n": n, "k": K, "dim": DIM, "instructions": [list(x) for x in ALL], "null_caption": NULL_CAPTION})
    jev_p = {i: np.load(f"{OUT}/jev/{i}.npy") for i in Q if os.path.exists(f"{OUT}/jev/{i}.npy")}
    llm_a = {i: json.load(open(f"{OUT}/llm/{i}.json")) for i in Q if os.path.exists(f"{OUT}/llm/{i}.json")}
    q16 = lambda x, lo, hi: np.clip(np.round((x - lo) / (hi - lo + 1e-12) * 65535), 0, 65535).astype(np.uint16)
    models = [m for m in MODELS if os.path.exists(f"{OUT}/emb/{m}/none.npy")]
    for m in models:
        E0 = torch.from_numpy(np.load(f"{OUT}/emb/{m}/none.npy")).float().to(dev)
        for iid, *_ in ALL:
            key, f = f"{m}/{iid}", f"{OUT}/emb/{m}/{iid}.npy"
            if not os.path.exists(f) or (key in meta["files"] and os.path.exists(f"{W}/bin/{m}__{iid}.txt")):
                continue
            E = torch.from_numpy(np.load(f)).float().to(dev)
            lo, hi, dlo, dhi = 9.0, -9.0, 9.0, -9.0
            nb_i, nb_s = [], []
            for s in range(0, n, 1024):
                S = E[s:s + 1024] @ E.T
                r = torch.arange(S.shape[0], device=dev)
                S[r, r + s] = float("nan")
                lo, hi = min(lo, float(S.nan_to_num(9).min())), max(hi, float(S.nan_to_num(-9).max()))
                v, i = S.nan_to_num(-9).topk(K, dim=1)
                nb_i.append(i), nb_s.append(v)
                if iid != "none":
                    S0 = E0[s:s + 1024] @ E0.T
                    S0[r, r + s] = float("nan")
                    D = S - S0
                    dlo, dhi = min(dlo, float(D.nan_to_num(9).min())), max(dhi, float(D.nan_to_num(-9).max()))
            # compressed embeddings for arbitrary cells: uncentred SVD keeps dot products, int8 per dimension
            U, sv, _ = torch.linalg.svd(E, full_matrices=False)
            P = U[:, :DIM] * sv[:DIM]
            scale = P.abs().amax(0) / 127
            Pq = torch.round(P / scale).clamp(-127, 127)
            Pd = Pq * scale
            pr = torch.randint(0, n, (2, 20000), device=dev)
            err = float(((Pd[pr[0]] * Pd[pr[1]]).sum(1) - (E[pr[0]] * E[pr[1]]).sum(1)).abs().mean())
            parts = [Pq.to(torch.int8).cpu().numpy(), torch.cat(nb_i).cpu().numpy().astype(np.uint16),
                     q16(torch.cat(nb_s).cpu().numpy(), lo, hi)]
            info = {"lo": lo, "hi": hi, "scale": scale.cpu().tolist(), "cell_abs_err": err}
            if iid != "none":
                info.update({"dlo": dlo, "dhi": dhi})
                af = f"{OUT}/emb/{m}/{iid}.anchor.npy"
                app = E.cpu().numpy() @ np.load(af).astype(np.float32)[0] if os.path.exists(af) else None
                if app is not None:  # similarity to the null caption: high = says nothing about u
                    parts.append(q16(app, -1, 1))
                    info.update({"app": True, "anchor_lo": float(app.min()), "anchor_hi": float(app.max())})
                if iid in jev_p and app is not None:
                    na = jev_p[iid] < 0.5
                    if 0 < na.sum() < n:
                        info["auroc_anchor_vs_jev"] = float(roc_auc_score(na, app))
                if iid in llm_a:
                    na = np.array([bool(NOT_MENTIONED.search(a)) for a in llm_a[iid]])
                    if 0 < na.sum() < n and app is not None:
                        info["auroc_anchor_vs_llm"] = float(roc_auc_score(na, app))
                    if 0 < na.sum() < n and iid in jev_p:
                        info["auroc_jev_vs_llm"] = float(roc_auc_score(na, -jev_p[iid]))
            # artifacts serve no binary types: base64 text, decoded in the page
            raw = b"".join(np.ascontiguousarray(p).tobytes() for p in parts)
            open(f"{W}/bin/{m}__{iid}.txt", "w").write(base64.b64encode(raw).decode())
            meta["files"][key] = info
            json.dump(meta, open(meta_f, "w"))
            print(key, f"lo {lo:.3f} hi {hi:.3f} cell err {err:.4f}", flush=True)
    # per-caption extras
    if jev_p:
        order = [i for i in Q if i in jev_p]
        raw = np.stack([np.round(jev_p[i] * 255).astype(np.uint8) for i in order]).tobytes()
        open(f"{W}/bin/jev.txt", "w").write(base64.b64encode(raw).decode())
        meta["jev_order"] = order
    for m in ("inbedder7b", "inbedder7b_na"):
        for iid in Q:
            src = f"{OUT}/gen/{m}/{iid}.json"
            if os.path.exists(src):
                os.makedirs(f"{W}/gen/{m}", exist_ok=True)
                open(f"{W}/gen/{m}/{iid}.json", "w").write(open(src).read())
    for iid, a in llm_a.items():
        json.dump(a, open(f"{W}/llm/{iid}.json", "w"))
    meta["models"] = models
    meta["gens"] = {m: [i for i in Q if os.path.exists(f"{W}/gen/{m}/{i}.json")] for m in ("inbedder7b", "inbedder7b_na")}
    meta["llm"] = list(llm_a)
    json.dump(meta, open(meta_f, "w"))
    json.dump(rows, open(f"{W}/captions.json", "w"))
    print("export done", len(meta["files"]), "files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "embed", "gen", "jev", "llm", "export", "audio"])
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--fresh", action="store_true", help="export: recompute every file (e.g. once new N/A labels exist)")
    a = ap.parse_args()
    if a.stage == "sample":
        sample()
    else:
        globals()[a.stage](a)
