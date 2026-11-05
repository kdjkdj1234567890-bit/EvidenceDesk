# -*- coding: utf-8 -*-
"""EvidenceDesk core: document chunking, evidence retrieval, Meta Llama answer
composition with mandatory citations, and an extractive fallback.

Design rule: the language model never invents content. It may only select and
compress spans from retrieved passages. Every sentence the user sees carries a
citation marker that maps to a stored chunk; otherwise we fall back to a fully
extractive answer built by deterministic code.
"""
import json, math, re, subprocess, os

LLAMAFILE = os.environ.get("ED_LLAMAFILE", os.path.expanduser("~/evidencedesk/llama3.2-1b.llamafile"))
MODEL_NAME = "Meta Llama 3.2 1B Instruct (local, llamafile)"
MAX_CTX = 1800  # stay well under the 2048 context

def load_docs(docdir):
    docs = []
    for fn in sorted(os.listdir(docdir)):
        if not fn.endswith(".txt"):
            continue
        raw = open(os.path.join(docdir, fn), encoding="utf-8").read()
        title = raw.splitlines()[0].strip()
        body = "\n".join(raw.splitlines()[1:]).strip()
        docs.append({"id": fn[:-4], "title": title, "body": body})
    return docs

def chunk_docs(docs, size=260, overlap=40):
    chunks = []
    for d in docs:
        paras = [p.strip() for p in re.split(r"\n\s*\n", d["body"]) if p.strip()]
        for i, p in enumerate(paras):
            while p:
                seg, p = p[:size], p[max(size - overlap, 1):]
                chunks.append({"doc": d["id"], "title": d["title"], "para": i, "text": seg})
    return chunks

def ngrams(s, n=2):
    s = re.sub(r"\s+", "", s)
    grams = [s[i:i+n] for i in range(max(len(s) - n + 1, 1))]
    grams += [s[i:i+3] for i in range(max(len(s) - 2, 1))]
    return grams

class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.df = {}
        for c in chunks:
            for g in set(ngrams(c["text"])):
                self.df[g] = self.df.get(g, 0) + 1
        self.N = max(len(chunks), 1)

    def _idf(self, g):
        return math.log(1 + self.N / (1 + self.df.get(g, 0)))

    _GENERIC_TERMS = {"문서", "예시", "데모", "샘플", "가상", "것", "경우", "관련",
                      "질문", "답변", "내용", "정보", "기준", "방법", "절차", "문의"}

    def distinctive_terms(self, query):
        """Content tokens of the query that are rare in this corpus. A token
        qualifies when it is 2+ chars, not a generic word, appears in the corpus,
        and appears in few chunks - so 'any document mentioning the topic' cannot
        masquerade as evidence for a question about a specific subject."""
        cap = max(1, -(-self.N // 7))  # ceil(N/7)
        terms = set()
        for tok in re.split(r"\s+", query):
            t = re.sub(r"(은|는|이|가|을|를|의|에|에서|으로|로|도|만|까지|부터|하고|와|과)$", "", tok)
            t = re.sub(r"(인가요|까요|나요|가요|궁금합니다|알려주세요|알고|싶어요|추천해줘|해주세요)$", "", t)
            if len(t) < 2 or t in self._GENERIC_TERMS or not re.search(r"[가-힣]", t):
                continue
            df = sum(1 for c in self.chunks if t in c["text"])
            if 1 <= df <= cap:
                terms.add(t)
        return terms

    def search(self, query, k=4):
        q = ngrams(query)
        scores = []
        for idx, c in enumerate(self.chunks):
            s = 0.0
            cg = ngrams(c["text"])
            cs = set(cg)
            for g in q:
                if g in cs:
                    s += self._idf(g) * (1 + cg.count(g) ** 0.5)
            s /= (len(c["text"]) ** 0.3 + 1e-9)
            scores.append((s, idx))
        scores.sort(reverse=True)
        top = [dict(self.chunks[i], rank=r + 1, score=round(s, 3))
               for r, (s, i) in enumerate(scores[:k]) if s >= 2.0]
        for r, c in enumerate(top):
            c["cite"] = r + 1
        return top

def build_prompt(query, passages):
    src = "\n".join(f"[{p['cite']}] ({p['title']}) {p['text']}" for p in passages)
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a retrieval filter. Given a Korean question and numbered source passages, "
        "output ONLY the numbers of the 1 or 2 passages that contain the answer, "
        "comma-separated (example: 1,3). No words, no explanation, numbers only. "
        "If no passage answers, output: none"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Question: {query}\n\nPassages:\n{src}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

def _llm_pick_sentence(query, text, timeout=60):
    """Second Llama pass: pick the answering sentence inside a chosen passage.
    Returns the verbatim sentence, or None to fall back to n-gram scoring."""
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    if len(sents) <= 1:
        return sents[0] if sents else text.strip()
    plist = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sents))
    prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You pick the sentence that answers the question. Reply with ONLY the sentence number. One digit. Nothing else."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Question: {query}\nSentences:\n{plist}\nWhich sentence answers the question?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n")
    try:
        out = subprocess.run(["/bin/sh", LLAMAFILE, "--cli", "--no-display-prompt", "-p", prompt,
                              "-n", "4", "--temp", "0.0", "--no-warmup"],
                             capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    m = re.search(r"\d+", out.stdout.strip())
    if not m:
        return None
    n = int(m.group(0))
    if not (1 <= n <= len(sents)):
        return None
    return sents[n - 1]

def _best_sentence(query, text):
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if len(s.strip()) > 5]
    if not sents:
        return text.strip()
    qg = set(ngrams(query))
    return max(sents, key=lambda s: len(set(ngrams(s)) & qg))


_STOPGRAMS = {"니다", "합니", "습니", "입니", "까요", "나요", "가요", "이요",
              "습니다", "합니다", "입니다", "인가요", "했습", "있습", "없습",
              "하는", "있는", "에서", "으로", "하는", "대한", "경우"}

def _content_grams(s):
    return set(g for g in ngrams(s) if g not in _STOPGRAMS)

def _question_type(query):
    if re.search(r"언제|기한|까지|날짜|시기", query):
        return "when"
    if re.search(r"몇\s*년", query):
        return "years"
    if re.search(r"몇\s*기", query):
        return "count"
    if re.search(r"얼마|수수료|세율|금액|요금|한도|금리|비용", query):
        return "money"
    return None

def _evidence_ok(query, sent, dg=None):
    """Relation-aware check: quoted sentence must share content n-grams with the
    query and, for when/value questions, actually carry that kind of fact.
    When distinctive query grams are supplied, the sentence must carry at least
    one - an amount/date from an unrelated document is not evidence."""
    if not (set(ngrams(sent)) & _content_grams(query)):
        return False
    if dg and not any(t in sent for t in dg):
        return False
    t = _question_type(query)
    if t == "when":
        return bool(re.search(r"\d+\s*(월|분기)|부터|까지|이내|이전|이후|요일|시점|\d+\s*(개월|일|주)\s*(전|후)", sent))
    if t == "years":
        return bool(re.search(r"\d+\s*년", sent))
    if t == "count":
        return bool(re.search(r"\d+\s*(기|회|개월|차)", sent))
    if t == "money":
        if re.search(r"수수료|비용|요금", query) and not re.search(r"수수료|비용|요금", sent):
            return False
        return bool(re.search(r"\d[\d,.]*\s*(%|원|만|억|천|배)", sent))
    return True

def _best_evidence_sentence(query, text, dg=None):
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if len(s.strip()) > 5]
    cands = [s for s in sents if _evidence_ok(query, s, dg)]
    if not cands:
        return None
    qg = set(ngrams(query))
    return max(cands, key=lambda s: len(set(ngrams(s)) & qg))

def llm_answer(query, passages, timeout=90, dg=None):
    """Meta Llama picks which retrieved passages hold the answer; deterministic
    code then quotes the best verbatim sentence from each picked passage.
    Returns None when the model's pick fails validation (caller falls back)."""
    if not passages or not os.path.exists(LLAMAFILE):
        return ("error", None)
    prompt = build_prompt(query, passages)
    if len(prompt) > MAX_CTX * 3:
        prompt = prompt[: MAX_CTX * 3]
    try:
        out = subprocess.run(
            ["/bin/sh", LLAMAFILE, "--cli", "--no-display-prompt", "-p", prompt,
             "-n", "12", "--temp", "0.0", "--no-warmup"],
            capture_output=True, text=True, timeout=timeout)
    except Exception:
        return ("error", None)
    if out.returncode != 0:
        return ("error", None)
    head = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    if "none" in head.lower():
        return ("none", None)
    picks = [int(x) for x in re.findall(r"\d+", head)]
    picks = [p for p in picks if 1 <= p <= len(passages)]
    picks = sorted(set(picks))[:2]
    if not picks:
        return ("error", None)
    qg = set(ngrams(query))
    parts, used = [], []
    for p in picks:
        text = passages[p - 1]["text"]
        sent = _llm_pick_sentence(query, text)
        # accept the model's sentence only when it passes the same
        # relation-aware evidence check; otherwise best passing candidate
        if sent is None or not _evidence_ok(query, sent, dg):
            sent = _best_evidence_sentence(query, text, dg)
        if sent is None:
            continue
        parts.append(f"{sent} [{p}]")
        used.append(p)
    if not parts:
        return ("none", None)
    # dedupe identical quoted sentences (overlapping chunks can repeat a sentence)
    seen, dparts, dused = set(), [], []
    for s, p in zip(parts, used):
        key = s.split(" [")[0]
        if key in seen:
            continue
        seen.add(key); dparts.append(s); dused.append(p)
    return ("ok", {"answer": " ".join(dparts), "citations": dused, "model": MODEL_NAME})

def extractive_fallback(query, passages, dg=None):
    """Deterministic answer: quote the best evidence sentences with citations."""
    out, cites = [], []
    for p in passages[:2]:
        best = _best_evidence_sentence(query, p["text"], dg)
        if best is None:
            continue
        out.append(f"{best.strip()} [{p['cite']}]")
        cites.append(p["cite"])
    if not out:
        return None
    seen, dout, dcites = set(), [], []
    for s, c in zip(out, cites):
        key = s.split(" [")[0]
        if key in seen:
            continue
        seen.add(key); dout.append(s); dcites.append(c)
    return {"answer": " ".join(dout), "citations": dcites,
            "model": "extractive fallback (no model output accepted)"}

REFUSAL = "등록된 문서에서 이 질문에 답할 근거를 확인하지 못했습니다. 관련 문서를 등록하거나 질문을 구체화해 주세요."

def answer(query, chunks):
    """Evidence-first answering with explicit refusal.
    - retrieval finds nothing solid          -> refuse
    - Llama says no passage answers ('none') -> refuse
    - Llama technical failure                -> extractive fallback only when
                                                retrieval confidence is high,
                                                otherwise refuse
    """
    tax = re.search(r"[가-힣]{2,}세(?=율|\s|기한|신고|은|는|이|가|$)", query)
    if tax:
        corpus = " ".join(c["text"] for c in chunks)
        if tax.group(0) not in corpus:
            return {"answer": REFUSAL, "citations": [], "passages": [],
                    "model": "none", "mode": "refused"}
    r = Retriever(chunks)
    dg = r.distinctive_terms(query) if _question_type(query) else set()
    passages = r.search(query)
    if not passages:
        return {"answer": REFUSAL, "citations": [], "passages": [],
                "model": "none", "mode": "refused"}
    if dg:
        grounded = [p for p in passages
                    if any(t in p["text"] for t in dg)]
        if not grounded:
            return {"answer": REFUSAL, "citations": [], "passages": passages,
                    "model": "none", "mode": "refused"}
        passages = grounded
        for i, p in enumerate(passages):
            p["cite"] = i + 1
    status, res = llm_answer(query, passages, dg=dg)
    if status == "ok":
        res["passages"] = passages
        res["mode"] = "llm"
        return res
    if status == "none":
        return {"answer": REFUSAL, "citations": [], "passages": passages,
                "model": MODEL_NAME, "mode": "refused"}
    if passages[0]["score"] >= 4.0:
        res = extractive_fallback(query, passages, dg=dg)
        if res is not None:
            res["passages"] = passages
            res["mode"] = "fallback"
            return res
    return {"answer": REFUSAL, "citations": [], "passages": passages,
            "model": "none", "mode": "refused"}
