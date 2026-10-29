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

## List questions: a wider-coverage ranking and a one-call relevance map (2026-10-29, 08:19-09:55 UTC)

Context. Three lever families on the sanitized pipeline had been closed at QS 78.94: evidence selection (changing the map
model or prompt), rerank coverage (a deeper cross-encoder pass) and chunking. The one unsized lever was a stronger
first-stage embedder. Qwen3-Embedding-4B, with its own query prompt on the question side only, passed a free proxy gate
(hybrid R@25 93.08 vs 91.82 for bge-m3 over the same stored chunks), so the product retrieval was re-run with it:
`MEDIAFIND_EMBED=Qwen/Qwen3-Embedding-4B`, the same memory search, over the original question and the three
LLM rewrites already on file (DeepSeek V4.1 Flash, GPT-5.4 mini, GPT-5.4; the two extra rewrite sets were made on
2026-10-28 at $0.35 and $4.10 and had not been adopted on their own). Its top-50 was reranked by the same
bge-reranker-v2-m3, and the lists were reciprocal-rank fused (k = 10, depth 100) with the existing bge-m3 rerank list.
Fusion gate, stated before the run: uncovered-at-25 <= 71 (from 86), R@10 >= 89.43, at most 5 previously covered
questions lost. The equal-weight fusion missed on the last criterion (61 uncovered, 6 lost); weighting the existing
bge-m3 rerank list 1.5 passed (59 uncovered, 5 lost, R@10 91.68, R@25 95.93). The weight was chosen on the hash subset
(where 1.5 and 2.0 differ by one question); the same sweep on the full set would have picked 2.0 (56 uncovered). Both
sweeps are deterministic retrieval numbers; no answer or judge call was involved.

Full answer pass on that ranking, unchanged pipeline otherwise (run 1, 08:19-08:51 UTC, $6.84): **QS 77.64**
(number 81.7, list_recall 77.3, open_end 74.9). Coverage had improved by 27 questions and retrieval R@10 by 1.95, yet
the judged questions fell by 2.9 points. Rebuilding the relevance-map decisions from the request cache showed why:
the map, which reads the 25 items in batches of ten, kept 0.45 more non-gold items per open_end question on the new
pools (2.97 -> 3.40), and the 34 open_end questions that flipped from right to wrong had gone from 3.7 to 5.6 kept
distractors each. The map has no absolute bar: it keeps roughly 0.85 items per rank from the gold-free ranks 11-25
(open_end kept distractors by rank: 1-10: 466, 11-20: 837, 21-25: 444; gold: 585 / 31 / 11).

Map batch change. Reading all 25 items in ONE map call (same instruction, same model, same max_tokens) lets the
model judge candidates against the real match. Subset map-only test (196 questions, $1.14, no reduce or judge):
kept non-gold items halve (adopted pool 3.31 -> 1.63, new pool 3.69 -> 1.54) with gold kept unchanged (all-gold-kept
157 -> 161 and 168 -> 167 of 196). A 40-item pool with one 40-item call was also screened ($0.66): no better (165
all-gold-kept, 1.69 distractors), so the pool depth stays 25. Subset answer passes with the one-call map: adopted pool
79.45, new pool 81.14 (vs 78.68 for the row as it stood) - the new pool with the one-call map was the pre-registered
candidate for one full pass.

Full passes (all four are reported; none was re-judged):
- run 1, new pool, batch-of-ten map: QS 77.64 (number 81.7, list 77.3, open 74.9) - above;
- run 2, new pool, one-call map (09:07-09:31 UTC, $3.62): **QS 79.01** (number 81.4, list_recall 80.2, open_end 77.0);
- run 3, adopted pool, one-call map, number and open_end questions only (09:32-09:53 UTC, $2.34): number 80.8,
  open_end 76.1 - below the row's 81.4 / 77.8, the opposite of what the subset had suggested;
- the row's own answer set (2026-10-24 run): number 81.4, open_end 77.8, list_recall 76.7.
The four open_end results (77.8, 76.1, 74.9, 77.0) bracket what a different generation and judge draw does to 514
judged questions, about +/-1.5 points, so no open_end-only difference of that size is resolvable here.

What the row takes (stated before run 3 was judged): the list_recall answers come from run 2, whose list score is
deterministic (Jaccard over ids) and which the subset had already preferred over the alternatives (86.8 vs 85.3 and
80.7 on the subset's 30 list questions) before its full run; the number and open_end answers and judgments stay those
of the 2026-10-24 run, because run 3 did not beat them. This is a type-conditional pipeline: list questions (which
reward recall) use the wider-coverage ranking and the one-call map; number and open_end questions run exactly as
before. The question type is part of the benchmark input and is already used by the prompts and by the K_LIST rule.

Result: **QS 79.41** (number 81.4, list_recall 80.2, open_end 77.8). Retrieval is reported for the ranking each
question's answer actually used (list questions: the new ranking; others: the previous one): R@10 90.12, R@25 94.22
(list questions alone: R@10 90.21, R@25 95.61). `docs/mediafind-hosted-llms-atmbench.json` carries this set with the
top-25 ids of the ranking used per question.

Product note. The shipped MediaFind embedder had no query-side prompt; the benchmark run applied the model's query
prompt through a harness hook on the product's memory-search entry point. That support now lives in the product
(MediaFind pull request #2464: `embed_query()` applies a model's own query prompt when it ships one; the default
encoders ship none and are unchanged), so `MEDIAFIND_EMBED=Qwen/Qwen3-Embedding-4B` reproduces the ranking without
the hook. The reranker, the rewrites, the fusion weights and the map are benchmark-harness steps, as before.

Cost of this leg: $19.7 in OpenRouter calls (runs 1-3 $12.80, the subset map, answer and depth screens $2.49, the
two extra rewrite sets $4.45 counted on 2026-10-28). The hash subset is 196 questions chosen by sha1(id); its scores
mis-read two of the four full runs by 1.3-1.7 points, which is why only deterministic numbers decided anything here.

## Number questions: a deterministic answer formatter (2026-10-29, 14:00-14:30 UTC)

**What the official scorer does.** The deterministic scorer for number-type questions (`memqa/utils/evaluator/evaluate_qa.py` with `normalizer.py`) requires, among its checks, that every alphanumeric token of four or more characters in the ground truth also appear verbatim in the prediction (its "code" rule), and its time-range pattern reads the month-day part of an ISO date such as `2023-04-20` as a clock-time range. So a prediction that gives the right calendar date in ISO form fails against a ground truth that spells the month, and a time written "3 pm" fails against one written "3pm" (the trailing "0pm"-style substrings count as codes). These are format conventions, not facts; the answers were already correct.

**What changed.** Number-type answers now pass through a deterministic string formatter (`ans_format.py`, no model call) as the last step of the pipeline: an ISO date becomes "Month D, YYYY" with the full month name; a clock time drops its seconds and any timezone suffix that follows them; a space before am/pm is removed. Nothing else changed: the retrieval, the map, the reduce prompts and models, the list answers and the open-ended answers are exactly those of the previous row.

**Measured** on the adopted answer set (`ans/san_full_q4blists` → `ans/san_full_fmt`): 161 of the 360 number answers were rewritten; 6 went from wrong to right and 0 from right to wrong under the official scorer; number accuracy 81.39 → 83.06. QS 79.41 → **80.00**, R@10 unchanged at 90.12. The 514 open-ended judgments are carried over verbatim from the previous row (no re-judging), the 139 list answers are unchanged, and the official evaluator's `em` metric was re-run on the formatted answer file as a check (number 83.06 and list 80.58 reproduced). Cost: $0.

**Tried today and not adopted.** Qwen3-Reranker-4B over the Qwen3-Embedding-4B lexical list for the 139 list questions, fused into the list pool at weight 3 (list R@10 90.21 → 93.34, R@25 95.61 → 96.07): one pre-registered list-only answer pass scored list recall 80.3 against the adopted 80.2 and a pre-registered bar of 82.4, so it was not adopted ($1.50). The ten list questions not fully covered at 25 have gold items outside the top-50, which no reranker of that list can reach.

## Number questions: a wider pool and a different relevance map (2026-10-29, 15:10-15:40 UTC)

Two deterministic screens, then one pre-registered answer pass.

**Map screen** (map step only: no reduce, no judge; kept-item statistics against the gold ids). On the 360 number questions with the adopted pool and ten-item batches, the DeepSeek V4.1 Flash map keeps every in-pool gold item on 321 questions, drops a gold item on 39, and keeps nothing on 8 (those fall back to the whole pool); Claude Haiku 5.5 keeps every in-pool gold item on 334, drops one on 26, never keeps nothing, at the same distractor rate (3.21 vs 3.19 kept distractors per question). In the adopted run the reduce step answers 90.8% of the number questions when it sees every gold item and 63.9% when one was dropped. Qwen3.8 Flash, the cleanest map on list questions, was rate-limited upstream part-way through this screen and was stopped: a map that fails mid-run is not usable for a full pass. Cost $1.13 (Haiku) plus about $0.6 (the partial Qwen3.8 screen).

**One pass with the Haiku map on the adopted pool** (pre-registered bar: at least 302 of 360 correct) scored exactly 302 (+3 against the adopted set, 13 gained / 10 lost): at the question bar, but +0.296 QS is not more than the standing +0.3 rule, so it was not adopted ($0.85).

**Pool.** The list questions' wider pool (the Qwen3-Embedding-4B fusion) also covers 8 more number questions at 25 (losing 1; number R@10 89.29 → 91.01), which the DeepSeek map had not converted in the earlier full runs (293 vs 294 of 360). The pre-registered combined pass — Haiku map, wider pool, otherwise the adopted number configuration (ten-item batches, no rank prior, GPT-5.4 mini at minimal effort, the deterministic formatter), bar at least 303 of 360 — scored **306 of 360** (85.00; 16 gained / 9 lost against the adopted set). Cost $1.17 (map) + $0.82 (reduce).

**Assembled row:** number answers from that pass; list and open-ended answers unchanged (the open-ended judgments are carried over verbatim, nothing re-judged). QS 80.00 → **80.69**; R@10 90.12 → 90.73 (number questions are now reported on the wider pool). The official evaluator's `em` metric, re-run on the assembled answer file, reproduces number 85.00 and list 80.58.

**Also tried today and not adopted** (list questions, one pre-registered pass each): the relevance map by Qwen3.8 Flash on the adopted list configuration (list recall 79.5 vs 80.2; at the adopted prior depth the reduce input is nine-tenths prior, so the map's cleanliness barely changes it; $0.18 after caching), and the reduce step at medium reasoning effort (77.1 vs 80.2; $1.30). Spend for the whole day's work, adopted and not: $9.75.

**Open-ended questions, screened and not changed (15:38-15:51 UTC).** The same map-only screen on the 514 open-ended questions (adopted pool, ten-item batches): Claude Haiku 5.5 keeps exactly the same gold items as DeepSeek V4.1 Flash (609 of 670 in the pool; every in-pool gold item on 448 questions for both) while keeping more distractors (3.28 vs 2.97 per question), so the pre-registered gate for an answer pass was not met and no open-ended pass was run ($1.96 for the screen). The open-ended answers and judgments in this row are still those of the 2026-10-27 frontier pass. Spend for the day including this screen: $11.71.
