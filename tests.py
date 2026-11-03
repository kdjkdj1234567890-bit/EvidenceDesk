# -*- coding: utf-8 -*-
import os, sys, time, unittest
sys.path.insert(0, os.path.expanduser("~/evidencedesk"))
import rag

DOCS = rag.load_docs(os.path.expanduser("~/evidencedesk/docs"))
CHUNKS = rag.chunk_docs(DOCS)

class TestCore(unittest.TestCase):
    def test_docs_loaded(self):
        self.assertEqual(len(DOCS), 4)
        self.assertGreater(len(CHUNKS), 8)

    def test_retrieval_tax(self):
        r = rag.Retriever(CHUNKS)
        top = r.search("부가가치세 신고 기한이 언제까지인가요?")
        self.assertTrue(top)
        self.assertEqual(top[0]["doc"], "semu-singo")

    def test_retrieval_loan(self):
        r = rag.Retriever(CHUNKS)
        top = r.search("대출 한도와 금리는?")
        self.assertEqual(top[0]["doc"], "jeongchaek-jageum")

    def test_retrieval_lease(self):
        r = rag.Retriever(CHUNKS)
        top = r.search("계약갱신요구권은 몇 년까지 가능한가요?")
        self.assertEqual(top[0]["doc"], "imdae-gyeyak")

    def test_no_match(self):
        res = rag.answer("우주 정거장 도킹 절차는?", CHUNKS)
        self.assertEqual(res["citations"], [])
        self.assertIn("확인하지 못했습니다", res["answer"])

    def test_fallback_cites(self):
        r = rag.Retriever(CHUNKS)
        p = r.search("주휴수당 지급 대상은?")
        fb = rag.extractive_fallback("주휴수당 지급 대상은?", p)
        self.assertTrue(fb["citations"])
        self.assertIn("[1]", fb["answer"])

    def test_llm_rejects_garbage(self):
        status, _ = rag.llm_answer("질문", [], timeout=5)
        self.assertEqual(status, "error")

    def test_llm_live_answer(self):
        if not os.path.exists(rag.LLAMAFILE):
            self.skipTest("no llamafile")
        r = rag.Retriever(CHUNKS)
        p = r.search("부가가치세 신고 기한은?")
        t0 = time.time()
        status, res = rag.llm_answer("부가가치세 신고 기한은?", p, timeout=110)
        el = time.time() - t0
        print(f"\n[live llm] {el:.1f}s -> {status} {res}")
        self.assertEqual(status, "ok")
        self.assertTrue(res["citations"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
