# -*- coding: utf-8 -*-
"""Held-out counterexample set (v2 corpus) - written AFTER the relation-aware
filter; never used during development. Run: python3 heldout.py"""
import os, sys, json
sys.path.insert(0, os.path.expanduser("~/evidencedesk"))
import rag

DOCS = rag.load_docs(os.path.expanduser("~/evidencedesk/docs"))
CHUNKS = rag.chunk_docs(DOCS)

CASES = [
    ("법인세 신고 기한이 언제인가요?", "refuse", "corpus has no 법인세 content"),
    ("종합소득세율이 어떻게 되나요?", "refuse", "tax-rate variant, absent"),
    ("사업자등록 발급 비용이 궁금합니다", "refuse", "fee paraphrase of the known residual"),
    ("희망드림 자금 수수료는 얼마인가요?", "refuse", "term-adjacent fee, absent"),
    ("임대료 연체 시 위약금은 얼마인가요?", "refuse", "corpus has 갱신 거절 사유, no 위약금"),
    ("주휴수당 지급 시기는 언제인가요?", "refuse", "corpus has 조건/계산법, no 지급 시기"),
    ("가상거래세 2기 확정 신고는 언제까지인가요?", "semu-singo", "control: 2월 10일"),
    ("희망드림 자금은 상시근로자 몇 명 미만이 대상인가요?", "jeongchaek-jageum", "control: 7인 미만"),
]

rows, ok = [], 0
for q, expect, note in CASES:
    res = rag.answer(q, CHUNKS)
    refused = res["mode"] == "refused"
    if expect == "refuse":
        good = refused
    else:
        cited = [res["passages"][c - 1]["doc"] for c in res["citations"]] if res["citations"] else []
        good = (not refused) and expect in cited
    ok += good
    rows.append({"q": q, "expect": expect, "mode": res["mode"],
                 "answer": res["answer"][:200], "ok": good, "note": note})
    print(("PASS" if good else "FAIL"), res["mode"], "|", q, "|", note)
json.dump({"passed": ok, "total": len(CASES), "rows": rows},
          open(os.path.expanduser("~/evidencedesk/heldout-results-v2.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"held-out: {ok}/{len(CASES)}")
