# LatentBridge — 3-minute demo video script

Judges weigh: Impact 30% · Meta AI tooling 25% · Demo 25% · Technical depth 20%. Show Muse on screen early
and often. Record at 1080p, dark terminal, large font. Fill the [brackets] with final numbers.

| Time | Screen | Voice-over |
|---|---|---|
| 0:00–0:15 | Title card: "LatentBridge — small Llama agents that talk in vectors". Laptop + "RTX 3050, 4 GB". | "Small AI models on a laptop can work as a team. But today they talk in text: one model writes a message, word by word, and the next one reads it. That writing is the slow part." |
| 0:15–0:35 | Simple diagram from the README (Muse Spark on top, 1B → bridge → 3B). | "LatentBridge replaces the written message with vectors. The reader runs once, a small trained bridge maps its hidden states into the solver's input space, and the reader never writes a word. Both Llamas stay frozen; only a 17-million-parameter bridge trains." |
| 0:35–1:15 | `python demo_cli.py --example 3`: Muse Spark plan → text channel (1B typing, 2.7 s) vs latent channel (0.06 s) → Muse Spark verify. | "Muse Spark plans the task and splits it. Watch the two channels: the text reader takes 2.7 seconds to write its notes; the latent reader takes 0.06. Then Muse Spark checks the answer." |
| 1:15–1:40 | `python chat_app.py`: ask the laptop-sale example on "Trained bridge", open "see each word". Then a wrong answer → red "Muse Spark rejected it", "Corrected answer", "saved as a lesson". | "Here it is live. Each vector the bridge sends is closest to the word it stands for. And when the local team gets it wrong, Muse Spark rejects the answer, writes a correction, and saves it as a lesson." |
| 1:40–2:05 | README "Muse Spark improver" table: 300 new problems, team 59% → Muse labels 96%, 115 of 123 wrong answers fixed. | "Those lessons replace the answer key. On 300 new problems, Muse Spark checked every answer the local team gave. It caught 115 of the 123 wrong ones and fixed them, so the labels went from 59 to 96 percent correct. One round of training on them didn't move accuracy yet; that's the next step." |
| 2:05–2:25 | (if run) `results/glimmer_eval.md`: Glimmer 30B none / text / latent / context. | "The same channel reaches Meta's open-weight Muse Glimmer 30B, which uses a different tokenizer, with zero training: [latent]% vs [text]% for the text message." |
| 2:25–2:50 | README results: 500-problem table (70.4% vs 60.2%, p = 1.6e-5) + size curve (60% vs 20–25% at 20 positions). | "On 500 test problems, the latent channel beats the text message, 70 vs 60 percent, and at the same message size it keeps three times more of the problem. It's not perfect: exact long numbers can slip a digit, and the agent checks for that." |
| 2:50–3:00 | `python agent/cli.py "Who first described tardigrades, and in what year?"` final answer, repo URL. | "Muse Spark as the brain, small local models doing the reading, talking in vectors. LatentBridge — open source, link below." |

Checklist before recording: `chat_app.py` running with MODEL_API_KEY set (Muse check on); `demo_cli.py --list`
works; close other GPU apps. Keep a pre-recorded fallback of the live parts.
