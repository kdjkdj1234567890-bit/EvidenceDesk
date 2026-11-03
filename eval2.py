# -*- coding: utf-8 -*-
"""3-arm blind ablation: A=retrieval only, B=+Llama passage pick, C=+sentence pick+validation.
Metrics: cited-doc accuracy, cited-sentence keyword hit, refusal accuracy."""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser("~/evidencedesk"))
import rag

DOCS = rag.load_docs(os.path.expanduser("~/evidencedesk/docs"))
CHUNKS = rag.chunk_docs(DOCS)

CASES = [
    ("가상거래세 1기 확정 신고는 언제까지인가요?", "semu-singo", "8월 10일"),
    ("간이납세자 기준 금액이 궁금합니다", "semu-singo", "6천만"),
    ("간이납세자도 전자세금계산서를 발급해야 하나요?", "semu-singo", "발급 의무는 없"),
    ("매출 누락 시 어떤 가산부담금이 나오나요?", "semu-singo", "가산부담금"),
    ("한세움에서 뭘 조회할 수 있나요?", "semu-singo", "한세움"),
    ("희망드림 자금 지원 대상 상시근로자 기준은?", "jeongchaek-jageum", "7인 미만"),
    ("대출 한도가 얼마인가요?", "jeongchaek-jageum", "9천만"),
    ("예시 금리가 어떻게 되나요?", "jeongchaek-jageum", "2.9%"),
    ("1분기 신청 접수 기간은 언제인가요?", "jeongchaek-jageum", "2월 1일"),
    ("신청에 필요한 서류는?", "jeongchaek-jageum", "사업자등록증"),
    ("갱신요구는 최대 몇 년인가요?", "imdae-gyeyak", "8년"),
    ("임대료를 몇 기분 연체하면 갱신을 못 하나요?", "imdae-gyeyak", "2기분"),
    ("보증금 반환 의무는 누구에게 있나요?", "imdae-gyeyak", "임대인"),
    ("권리금은 언제부터 보호되나요?", "imdae-gyeyak", "3개월"),
    ("주휴수당 지급 대상 조건은?", "goyong-bosom", "12시간"),
    ("주휴수당 계산 방법은?", "goyong-bosom", "6일"),
    ("근로자안심보험 자격 취득 신고 기한은?", "goyong-bosom", "10일"),
    ("이 데모 문서의 예시 시급 기준은 얼마인가요?", "goyong-bosom", "10,500원"),
    ("우주 정거장 도킹 절차를 알려주세요", None, None),
    ("점심 메뉴 추천해줘", None, None),
    ("비트코인 시세 전망은?", None, None),
    ("사업자등록증 발급 수수료는 얼마인가요?", None, None),
    ("육아휴직 급여는 얼마인가요?", None, None),
    ("법인세율이 궁금합니다", None, None),
]

def arm_a(q, passages):
    if not passages: return None
    p = passages[0]
    return {"doc": p["doc"], "sent": rag._best_sentence(q, p["text"])}

def arm_b(q, passages):
    if not passages: return None
    status, res = rag.llm_answer(q, passages)   # passage pick + sentence pick = C behavior
    # arm B = passage pick only: emulate by running passage pick then n-gram sentence
    # -> reuse llm pick stage via monkey-level split is complex; approximate B by
    #    taking llm's picked passage ids from prompt stage only.
    return None  # placeholder; B computed inline below

def run(arm):
    r = rag.Retriever(CHUNKS)
    rows = []
    for q, edoc, kw in CASES:
        passages = r.search(q)
        if arm == "A":
            res = arm_a(q, passages)
            out = {"mode": "llm" if res else "refused",
                   "doc": res["doc"] if res else None,
                   "sent": res["sent"] if res else ""}
        elif arm == "B":
            if not passages:
                out = {"mode": "refused", "doc": None, "sent": ""}
            else:
                st, res = _passage_pick_only(q, passages)
                if st != "ok":
                    out = {"mode": "refused" if st == "none" else "refused", "doc": None, "sent": ""}
                else:
                    p = passages[res[0] - 1]
                    out = {"mode": "llm", "doc": p["doc"], "sent": rag._best_sentence(q, p["text"])}
        else:  # C
            full = rag.answer(q, CHUNKS)
            if full["mode"] == "refused" or not full["citations"]:
                out = {"mode": "refused", "doc": None, "sent": ""}
            else:
                p = full["passages"][full["citations"][0] - 1]
                out = {"mode": "llm", "doc": p["doc"], "sent": full["answer"]}
        rows.append({"q": q, "expect": edoc, "kw": kw, **out})
    return rows

def _passage_pick_only(q, passages):
    """Arm B core: Llama picks passages, deterministic sentence scoring."""
    import re, subprocess
    prompt = rag.build_prompt(q, passages)
    try:
        out = subprocess.run(["/bin/sh", rag.LLAMAFILE, "--cli", "--no-display-prompt",
                              "-p", prompt, "-n", "12", "--temp", "0.0", "--no-warmup"],
                             capture_output=True, text=True, timeout=90)
    except Exception:
        return ("error", None)
    if out.returncode != 0:
        return ("error", None)
    head = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    if "none" in head.lower():
        return ("none", None)
    picks = sorted({int(x) for x in re.findall(r"\d+", head) if 1 <= int(x) <= len(passages)})[:2]
    return ("ok", picks) if picks else ("error", None)

def score(rows):
    ans = [x for x in rows if x["expect"]]
    una = [x for x in rows if not x["expect"]]
    return {
        "doc_acc": round(sum(x["doc"] == x["expect"] for x in ans) / len(ans), 3),
        "sent_acc": round(sum(x["kw"] in x["sent"] for x in ans) / len(ans), 3),
        "refusal_acc": round(sum(x["mode"] == "refused" for x in una) / len(una), 3),
    }

if __name__ == "__main__":
    arm = sys.argv[1]
    rows = run(arm)
    s = score(rows)
    print(f"ARM {arm}: {json.dumps(s, ensure_ascii=False)}")
    json.dump({"summary": s, "rows": rows},
              open(os.path.expanduser(f"~/evidencedesk/eval2-v2-arm{arm}.json"), "w"),
              ensure_ascii=False, indent=1)
