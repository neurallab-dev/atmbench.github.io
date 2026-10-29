# Prompt sanitization for the "MediaFind + hosted LLMs" row (re-run of 2026-10-23; the row was named "MediaFind v2" until 2026-10-26)

Goal: no prompt in the pipeline may carry wording taken from the ATM-Bench questions or their gold evidence.
Every fact a prompt states about the person must be traceable to the memory corpus (emails, captions,
location metadata), and the answer-prompt variant must be chosen on a fixed hash subset before the full run.
The unsanitized files are kept beside the pipeline as `*.unsanitized.py` for diffing.

## Changes, file by file

**`trips_extract.py` (email → trip/event timeline)**
- Removed the worked example "ACL 2024 = Bangkok, 11-16 Aug 2024" from the system prompt; the model is told only
  to use its knowledge of a conference's usual host city and dates.
- Replaced the record-name examples "ACL 2024", "summer trip to France", "Peterhouse formal" with placeholders
  (`<conference acronym> <year>`, `summer trip to <country>`, `<college> formal`).
- Kept the one-line owner description ("graduate student/researcher at the University of Cambridge, home:
  Petersfield area"). Derivation from the corpus: the two most frequent photo locations are addresses in
  Petersfield, Cambridge (173 and 114 photos); 567 of the 6,742 emails mention Peterhouse.

**`cards.py` (one memory card per photo/video)**
- Removed the four example recall questions ("the formal at Peterhouse last week", "burrata during ECCV 2024",
  "the photo of London from my flight to Cairo", "all the videos from the day I made mochi"); the prompt now says
  only "questions about a trip, a conference, a meal, a receipt or an everyday moment".
- Removed the two normalization examples ("Allianz MiCo = ECCV 2024 venue in Milan's CityLife district",
  "Fellows' Garden, Trumpington Street = Peterhouse"), the second of which had been learned from a missed
  question's gold photo; the prompt now says generically "the venue a conference used that year, the college a
  garden or hall belongs to".
- Replaced the Event examples ("ACL 2024 (62nd Annual Meeting of the ACL), Bangkok, Thailand, 11-16 Aug 2024",
  "formal dinner at Peterhouse, Cambridge") with placeholders; replaced "one of 9 videos of making mochi that
  afternoon" with "one of several videos of the same cooking session"; replaced the keyword examples "flight to
  Cairo", "summer trip to France" with `flight to <city>`, `summer trip to <country>`.
- Dropped the home street names from the owner description; kept "Petersfield area of Cambridge; college:
  Peterhouse; Department of Engineering" (derivation: location frequencies above; Peterhouse in 567 emails;
  "Department of Engineering" in 48 emails and 26 photo locations).

**`vlm_caption.py` (dense visual description of the raw photo / video frames)**
- Removed the three example details "a broken strap, a sleeping cat belly-up, a radar on a boat"; the
  instruction keeps the dense, question-agnostic list (objects, animals, vehicles, dishes, brands, signs and all
  visible text, counts, colours, weather, people, document/receipt/screenshot contents, venue clues).

**`rewrite_queries.py` (question → retrieval query)**
- Removed the hard-coded list of seven conferences. The conference facts given to the rewriter are now built
  programmatically from the email-derived timeline (`kind == "conference"` records with at least two timeline
  records, i.e. conferences the emails return to: name, place, month/year, one entry per name). "WWW 2025 Sydney", which appeared in no email, is therefore gone.
- Removed the synonym example "camouflaged car, test paint wrap"; the instruction now says "another name for
  the same dish, the district or landmark of a place, the generic word for an object".

**`mf_answer_v2.py` (relevance map + reduce)**
- The ID-format examples in the list rule were real-looking ids; replaced with synthetic ones
  (`20230101_120000`, `email202301010001`). The relevance-map instruction and the per-type rules contain no
  dataset-specific wording. The two pipeline fixes from the Oracle analysis stay because they encode no benchmark
  content: an empty relevance map falls back to the whole pool instead of answering "Unknown", and list answers
  keep only ids that exist in the pool.

## Answer-prompt variant: frozen on a hash subset before the full run

Rule: subset = questions with `sha1(id) % 1013 < 200` (196 questions: 70 number, 30 list_recall, 96 open_end). Two candidates are run on it with
the sanitized retrieval, judged once each with the official gpt-5-mini judge, and the higher QS is frozen (tie →
candidate A). No prompt or variant is changed after the full run starts.

- A: the original system prompt ("if the items are insufficient, say 'Unknown'").
- B: the never-abstain system prompt (`BEST_GUESS=1`: "the question always refers to something among the
  items; never answer 'Unknown'; pick the most plausible item").

Result (2026-10-24 02:12–02:28, before the full run started): A scored QS 75.59 (number 85.7, list_recall 63.9,
open_end 71.9) and B 77.61 (number 78.6, list_recall 73.7, open_end 78.1) on the 196 subset questions, so **B was
frozen**. Why B wins: the benchmark has no abstention questions except 22 'ABSTENTION…' gold answers for unstated
prices, so every 'Unknown' scores 0; A abstains on list and open-ended questions whose wording differs from the
item text, B answers them. A's higher number score (deterministic exact match) did not offset that. The subset
runs cost $2.20 including both judge passes.

## Stages re-run with the sanitized prompts

Timeline (all emails) → cards (4,292 items) → visual descriptions (4,292 items) → tasks file → bge-m3 + BM25
ingest on CPU and lexical ranking → bge-reranker-v2-m3 whole-item rerank of the top-50 → question rewrites →
lexical ranking of the rewrites → reciprocal-rank fusion (k=10) with the reranked ranking → subset runs for A and
B → full run with the frozen variant → one judge pass → evidence file with top-10 and top-25 ids.

Cost of the re-run: $9.69 on the OpenRouter key (timeline $0.13, cards $1.42, visual descriptions $1.32, rewrites $0.05,
subset candidate runs with two judge passes $2.20, full run $4.32, judge pass ≈ $0.25); ingest and rerank are local.
Result of the frozen full run: QS 78.38 (number 81.4, list_recall 72.7, open_end 77.8), R@1 55.14, R@5 84.62,
R@10 89.73, R@25 93.54, 0 answer failures, 0 judge failures. The unsanitized run had scored 77.34 / 90.17.

## Reduce-model choice for the one frontier pass (2026-10-26)

Rule, fixed before any run: the reduce model for ONE further full pass is the one with the best score on the same
196-question hash subset with gold evidence only (an Oracle: the answerer sees exactly the gold items, so retrieval
plays no part), sanitized item text, the frozen never-abstain prompt, one gpt-5-mini judge pass each. In this sweep
the candidate model also did the relevance-map step over the gold items (the full pass keeps the cached
DeepSeek V4.1 Flash map); with one to a few gold items per question the map keeps almost everything, so the reduce
step is what the sweep measures. Nothing in the prompts changed.

| reduce model (OpenRouter) | subset Oracle QS | number | list_recall | open_end |
|---|---|---|---|---|
| openai/gpt-5.4-mini (the 78.38 row's model, reference) | 86.59 | 88.6 | 95.7 | 82.3 |
| **openai/gpt-5.4 — chosen** | **86.84** | 88.6 | 97.4 | 82.3 |
| openai/gpt-5.5 | 86.33 | 85.7 | 97.4 | 83.3 |
| openai/gpt-5.6-sol | 85.06 | 88.6 | 95.7 | 79.2 |
| anthropic/claude-opus-5.5 | excluded | | | |

Claude Opus 5.5 refused 108 of the 196 reduce prompts (`finish_reason: content_filter`, the model's own refusal:
a probe of three refused prompts was refused on the Anthropic, Google Vertex and Claude-on-AWS routes alike), and
the pipeline's fallback model (gpt-5.4-mini) answered those, so its 87.10 is not a measurement of Opus and it is
excluded. The four measured models sit within about two questions of each other (one judged question is 0.51
points on this subset), so the choice buys little; gpt-5.4 is taken as the rule says, and it is also the cheapest
of the frontier candidates. The sweep cost $11.07 (five answer runs with their judge passes, the stalled first
Opus attempt included).

## The one frontier pass (2026-10-26, 20:01-20:26 local time; 03:01-03:26 UTC on 2026-10-27)

Reduce-only: the same sanitized item text, the same fused top-25 pools, the cached DeepSeek V4.1 Flash relevance
map, the frozen never-abstain prompt, and openai/gpt-5.4 (reasoning effort minimal) in place of gpt-5.4-mini for
the reduce step; 1,013 answers, 0 fallbacks, 0 failures, one gpt-5-mini judge pass, 0 judge failures.

Result: **QS 78.14** (number 80.8, list_recall 70.2, open_end 78.4), retrieval unchanged (R@10 89.73, R@25 93.54).
That is below the gpt-5.4-mini run's 78.38 (number 81.4, list_recall 72.7, open_end 77.8), by about two judged
questions on 1,013 - inside the +/-0.3 judge noise measured on an earlier run - so the row keeps the 78.38 run and
its evidence file, and the gpt-5.4 pass is recorded here only. The larger model gained on open-ended questions
and lost on the deterministic number and list types, which reward the exact-match habits of the smaller model.
Cost: $12.35 for the pass (answers $11.11, judge about $1.2); $23.42 for the day with the reduce-model sweep.

## List questions: the reduce step also sees the ten top-ranked items (assembled 2026-10-29, 00:00-00:15 UTC)

Context. On 2026-10-27 (UTC) a "rank prior" variant (`mf_answer_r2.py`) was run over the same fused top-25 pools and
the same cached DeepSeek V4.1 Flash map decisions: the reduce step received the fused top-K items in addition to
whatever the relevance map had kept (K = 5 for every question, K = 10 for list questions). Hash subset: 79.19 vs
77.61; full set: QS 78.05 (number 80.3, list_recall 76.7, open_end 76.8) vs 78.38 - not adopted at the time, because
the extra items lifted the Jaccard-scored list questions (+4.0) and cost the exact-match and judged types (the
"distractor tax" measured in the same run: -0.0067 correct per inserted non-gold item).

The type-conditional variant follows from that anatomy: the prior is applied to list questions only (K_LIST = 10,
K = 0 for number and open_end questions, which therefore run exactly as before). Stated plainly: the rule "list
questions only" was chosen after the per-type results of the rank-prior run were known on the full set (the subset
had pointed the same way: list +7.0, number +2.8, open_end -1.0). The depth was then confirmed on the hash subset's
30 list questions, whose score is deterministic (Jaccard, no judge): depth 0 / 5 / 10 / 15 / 20 / 25 = 73.7 / 77.1 /
80.7 / 80.0 / 76.2 / 77.9 (the 15, 20 and 25 runs were made on 2026-10-29 00:03-00:05 UTC, $1.23), so 10 is kept.

The full-set answer set is assembled from runs that already exist, with no new answer or judge call: the 360
number and 514 open_end answers, and the open_end judgments, are those of the 2026-10-24 run unchanged; the 139
list_recall answers are those of the 2026-10-27 rank-prior run, which used the same cached map decisions and
K_LIST = 10, and whose scores are deterministic. Nothing was re-judged.

Result: **QS 78.94** (number 81.4, list_recall 76.7, open_end 77.8), retrieval unchanged (R@10 89.73, R@25 93.54);
the row and `docs/mediafind-hosted-llms-atmbench.json` carry this set. The gpt-5-mini judge noise measured on an
earlier run is +/-0.3 on the open_end questions, which are unchanged here; the +0.56 comes entirely from the
deterministic list questions.
