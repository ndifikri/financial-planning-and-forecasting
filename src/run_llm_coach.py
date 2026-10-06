"""CLI: generate pesan AI Financial Coach (gpt-6-luna) dari output scoring.

Jalankan di mesin yang punya akses ke api.openai.com:
    pip install -r requirements.txt
    set OPENAI_API_KEY=sk-...        (atau isi file .env: openai-api-key=sk-...)
    python -m src.run_llm_coach --n 3            # 1 contoh per health band (default)
    python -m src.run_llm_coach --client 315     # nasabah tertentu
Output: artifacts/llm_examples.json
"""
import argparse
import json

import pandas as pd

from . import config as C
from . import llm_advisor as LLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", type=int, nargs="*")
    ap.add_argument("--out", default=str(C.ARTIFACT_DIR / "llm_examples.json"))
    a = ap.parse_args()
    plan = pd.read_csv(C.ARTIFACT_DIR / "client_financial_plan.csv")
    budget = pd.read_csv(C.ARTIFACT_DIR / "client_budget_next_month.csv")
    if a.client:
        picks = plan[plan.client_id.isin(a.client)]
    else:
        picks = pd.concat([g.sort_values("health_score").iloc[[len(g) // 2]] for _, g in plan.groupby("health_band")])
    results = []
    for _, r in picks.iterrows():
        payload = LLM.build_payload(r.to_dict(), budget[budget.client_id == r.client_id].to_dict("records"))
        advice, source = LLM.generate_advice(payload, use_llm=True)
        results.append({"band": r.health_band, "payload": payload, "advice": advice, "source": source})
        print(f"\n=== client {r.client_id} | {r.health_band} | source={source} ===")
        print(json.dumps(advice, ensure_ascii=False, indent=1))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
