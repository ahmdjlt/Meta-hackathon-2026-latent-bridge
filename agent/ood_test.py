"""Step 0 go/no-go: does the GSM8K-trained latent bridge carry general web text?

20 short encyclopedia-style passages, one factual question each (agent/ood_test.json).
Conditions: raw passage, text message from 1B, latent (stride 1, <=48-token chunks), no message.
Scoring: case-insensitive match of any accepted answer string in the 3B's reply.
Rule: continue with the latent reader if latent is within 15 points of the text message.

Usage: python agent/ood_test.py
"""
import json
from pathlib import Path

from reader import Reader

HERE = Path(__file__).resolve().parent
CONDS = ("raw", "text", "latent", "none")


def correct(reply, answers):
    r = reply.lower()
    return any(a.lower() in r for a in answers)


def main():
    items = json.loads((HERE / "ood_test.json").read_text(encoding="utf-8"))
    rd = Reader()
    print(f"models + bridge loaded in {rd.load_s:.1f} s; bridge meta: {rd.meta}")
    rows = []
    for it in items:
        row = {"id": it["id"], "passage_tokens": rd.ntok(it["passage"])}
        for c in CONDS:
            out = rd.answer(it["passage"], it["question"], channel=c)
            out["correct"] = correct(out["answer"], it["answers"])
            row[c] = out
        rows.append(row)
        print(f"{it['id']:<12} {row['passage_tokens']:>4} tok  " +
              "  ".join(f"{c}={'Y' if row[c]['correct'] else '-'}" for c in CONDS) +
              f"   latent: {row['latent']['answer'][:60]!r}")
    n = len(rows)
    summary = {}
    for c in CONDS:
        rs = [r[c] for r in rows]
        summary[c] = {"acc": sum(x["correct"] for x in rs) / n,
                      "reader_s": sum(x["reader_s"] for x in rs) / n,
                      "solver_s": sum(x["solver_s"] for x in rs) / n,
                      "total_s": sum(x["total_s"] for x in rs) / n,
                      "msg_positions": sum(x["msg_positions"] for x in rs) / n,
                      "msg_bytes": sum(x["msg_bytes"] for x in rs) / n,
                      "chunks": sum(x["chunks"] for x in rs) / n}
    gap = summary["text"]["acc"] - summary["latent"]["acc"]
    decision = "GO (latent reader)" if gap <= 0.15 else "NO-GO (text path only)"
    summary["latent_minus_text_points"] = round(-gap * 100, 1)
    summary["decision"] = decision
    summary["n"] = n
    (HERE / "ood_results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    lines = ["| Condition | Accuracy | Msg positions | Msg bytes | Reader | 3B | Total |", "|---|---|---|---|---|---|---|"]
    for c in CONDS:
        s = summary[c]
        lines.append(f"| {c} | {s['acc']*100:.0f}% ({round(s['acc']*n)}/{n}) | {s['msg_positions']:.0f} | "
                     f"{s['msg_bytes']:.0f} | {s['reader_s']:.2f} s | {s['solver_s']:.2f} s | {s['total_s']:.2f} s |")
    lines.append(f"\nLatent minus text: {-gap*100:+.0f} points -> {decision}")
    md = "\n".join(lines)
    (HERE / "ood_results.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
