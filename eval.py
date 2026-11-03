# -*- coding: utf-8 -*-
"""Ablation eval: retrieval-only vs retrieval+Llama selection.
Measures: correct cited document, refusal correctness, latency, fallback rate.
"""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser("~/evidencedesk"))
import rag

DOCS = rag.load_docs(os.path.expanduser("~/evidencedesk/docs"))
CHUNKS = rag.chunk_docs(DOCS)

# (question, expected doc id or None for "should refuse")
CASES = [
    ("가상거래세 1기 확정 신고는 언제까지인가요?", "semu-singo"),
    ("간이납세자 기준 금액이 궁금합니다", "semu-singo"),
    ("간이납세자도 전자세금계산서를 발급해야 하나요?", "semu-singo"),
    ("매출 누락 시 어떤 가산부담금이 나오나요?", "semu-singo"),
    ("한세움에서 뭘 조회할 수 있나요?", "semu-singo"),
    ("희망드림 자금 지원 대상 상시근로자 기준은?", "jeongchaek-jageum"),
    ("대출 한도가 얼마인가요?", "jeongchaek-jageum"),
    ("예시 금리가 어떻게 되나요?", "jeongchaek-jageum"),
    ("1분기 신청 접수 기간은 언제인가요?", "jeongchaek-jageum"),
    ("신청에 필요한 서류는?", "jeongchaek-jageum"),
    ("갱신요구는 최대 몇 년인가요?", "imdae-gyeyak"),
    ("임대료를 몇 기분 연체하면 갱신을 못 하나요?", "imdae-gyeyak"),
    ("보증금 반환 의무는 누구에게 있나요?", "imdae-gyeyak"),
    ("권리금은 언제부터 보호되나요?", "imdae-gyeyak"),
    ("주휴수당 지급 대상 조건은?", "goyong-bosom"),
    ("주휴수당 계산 방법은?", "goyong-bosom"),
    ("근로자안심보험 자격 취득 신고 기한은?", "goyong-bosom"),
    ("이 데모 문서의 예시 시급 기준은 얼마인가요?", "goyong-bosom"),
    ("우주 정거장 도킹 절차를 알려주세요", None),
    ("점심 메뉴 추천해줘", None),
    ("비트코인 시세 전망은?", None),
    ("사업자등록증 발급 수수료는 얼마인가요?", None),  # mentioned but fee not stated
    ("육아휴직 급여는 얼마인가요?", None),
    ("법인세율이 궁금합니다", None),
]

def run():
    r = rag.Retriever(CHUNKS)
    rows = []
    for q, expect in CASES:
        t0 = time.time()
        passages = r.search(q)
        # arm A: retrieval only
        a_doc = passages[0]["doc"] if passages else None
        a_ok = (a_doc == expect) if expect else (a_doc is None)
        # arm B: retrieval + Llama
        res = rag.answer(q, CHUNKS)
        el = time.time() - t0
        if expect is None:
            b_ok = res["mode"] == "refused"
            b_doc = None
        else:
            cited = [passages[c - 1]["doc"] for c in res["citations"]] if res["citations"] else []
            b_doc = cited[0] if cited else None
            b_ok = bool(cited) and expect in cited
        rows.append({"q": q, "expect": expect, "a_doc": a_doc, "a_ok": a_ok,
                     "b_doc": b_doc, "b_ok": b_ok, "mode": res["mode"], "ms": int(el * 1000)})
        print(f"[{'A:O' if a_ok else 'A:X'}|{'B:O' if b_ok else 'B:X'}|{res['mode']:8s}] {q[:30]}")
    ans = [x for x in rows if x["expect"]]
    una = [x for x in rows if not x["expect"]]
    a_acc = sum(x["a_ok"] for x in ans) / len(ans)
    b_acc = sum(x["b_ok"] for x in ans) / len(ans)
    a_ref = sum(x["a_ok"] for x in una) / len(una)
    b_ref = sum(x["b_ok"] for x in una) / len(una)
    modes = {}
    for x in rows:
        modes[x["mode"]] = modes.get(x["mode"], 0) + 1
    summary = {
        "answerable": len(ans), "unanswerable": len(una),
        "retrieval_only_doc_acc": round(a_acc, 3),
        "llama_pipeline_cited_doc_acc": round(b_acc, 3),
        "retrieval_only_correct_silence": round(a_ref, 3),
        "llama_pipeline_correct_refusal": round(b_ref, 3),
        "modes": modes,
        "avg_ms": int(sum(x["ms"] for x in rows) / len(rows)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    json.dump({"summary": summary, "rows": rows},
              open(os.path.expanduser("~/evidencedesk/eval-results-v2.json"), "w"),
              ensure_ascii=False, indent=1)

if __name__ == "__main__":
    run()
