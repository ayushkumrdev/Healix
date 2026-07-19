# Healix paper — honest publication roadmap

This is the straight story on getting `healix.tex` into an IEEE venue. I wrote
the manuscript and can revise it; I **cannot** submit it or make it accepted —
that is a months-long human peer-review process requiring a corresponding author,
an ORCID, an institutional/independent affiliation, and a signed IEEE copyright
form. Below is what is real, what is missing, and the concrete steps.

## 1. Novelty — honest assessment

- **Altitude-matched retrieval (RAPTOR × HyDE)** is the genuine novel hook.
  HyDE and RAPTOR are each well-known; generating hypotheticals at *two
  abstraction levels* and routing each to the matching index tier is, to my
  knowledge, not previously published. This is an **incremental-but-real**
  contribution — the kind that fits an applied conference or a journal like
  *IEEE Access*, not a top-tier ML venue on its own.
- **Agentic Mixture-of-Medical-Experts** is a solid systems/engineering
  contribution, weaker on novelty (routing-over-experts has precedent). It
  strengthens the paper but is not the headline.
- **Local-first, disclaimer-free medical assistant** is a good framing/systems
  story with privacy appeal.

Verdict: publishable at an **applied/health-informatics venue** *if* the
evaluation is strengthened (see §2). As-is it would likely be desk-rejected at a
selective venue for thin evaluation.

## 2. What must be added before it is submission-strong

**Update (July 19, 2026):** the paper now contains FOUR pre-registered
experiments with bootstrap CIs (`eval_retrieval.py`, `eval_keyword.py`,
`eval_broad.py`, `eval_latency.py`), and its identity has changed. It is now an
**honest empirical systems paper**:

- Retrieval augmentations (hybrid, HyDE, RAPTOR/altitude): **no win** on this
  corpus — dense is near ceiling (R@10 0.88–0.96); hybrid's keyword gap +0.006
  (CI includes 0, rescued 3 / lost 0); RAPTOR overview covers *fewer* subtopics
  than leaves at fixed budget (−0.33, CI [−0.46, −0.20], significant). All
  reported plainly.
- **One real, quantified win:** offloading HyDE to a 0.5B model cuts end-to-end
  TTFT 14.7s → 6.4s (2.3×) while the isolated call only drops 0.76s — the gap is
  GPU contention. "Auxiliary-model cost is set by contention, not parameter
  count" is a genuine, transferable systems finding.
- The earlier "3×" claim was a cold-start artifact; it has been corrected in the
  paper. Never ship the uncorrected version.

This framing ("we proposed X, tested it honestly, most of it didn't help on a
curated corpus, here is the rule for practitioners + a real latency finding") is
credible at IEEE Access, ICHI, or a good workshop — negative results with rigor
are publishable there. It will still not carry a top-tier ML venue. Remaining
gaps if you want to strengthen further:

1. **A harder retrieval set where the components can actually help** — the null
   result is on *in-domain single-target* recovery, where a strong reranker
   equalizes everything. Hybrid/HyDE are designed for out-of-distribution,
   term-heavy, or truly colloquial queries (rare drug/gene names, lay symptom
   descriptions). Build an OOD/term-heavy query set; if hybrid/HyDE still don't
   help, that is itself an honest, publishable negative result — but you must
   know which it is.
2. **A labeled broad-question set** to quantify the RAPTOR/altitude arm
   (currently only qualitative). This is where altitude-matching should win, and
   it is the paper's novelty — it needs a number.
3. **Answer-quality evaluation** — an LLM-as-judge or small human study rating
   grounded answer accuracy/faithfulness/helpfulness vs. no-RAG and flat-RAG.
4. **Baselines + statistics**: HyDE-only / RAPTOR-only ablations of the coupling,
   multiple seeds, confidence intervals.

I can build the answer-quality harness next if you want — it is the single
highest-leverage addition.

## 3. Realistic IEEE venues (fastest → most selective)

| Venue | Type | Fit | Notes |
|---|---|---|---|
| **IEEE Access** | Open-access journal | Good | Broad scope, faster (~4–8 wk first decision), APC applies. Most realistic first target. |
| **IEEE ICHI** (Int'l Conf. Healthcare Informatics) | Conference | Strong | Directly on-topic; annual deadlines. |
| **IEEE BigData / ICDM workshops** | Workshop | Good | Lower bar, fast, good for a first paper. |
| **IEEE EMBC** | Conference | Medium | Biomed engineering; large, applied. |
| **IEEE JBHI** (J. Biomedical & Health Informatics) | Journal | Aspirational | Higher bar; needs the answer-quality study. |

Post to **arXiv** first (cs.IR / cs.CL) to timestamp the idea — it is free,
immediate, and does not preclude IEEE submission (check the venue's preprint
policy; IEEE generally allows arXiv preprints).

## 4. Concrete submission steps (once evaluation is strengthened)

1. Finalize authorship + affiliations; every author needs an ORCID.
2. Fill real numbers into all tables (the retrieval table is auto-filled from the
   eval JSON; keep the answer-quality table honest).
3. Run the PDF through **IEEE PDF eXpress** (venue gives you a conference ID) for
   Xplore-compatibility.
4. Write a cover letter; suggest reviewers if the venue asks.
5. Submit via the venue portal (IEEE Access: ScholarOne; conferences: EDAS/CMT).
6. Sign the **IEEE electronic copyright form** on acceptance.
7. Expect one or more revision rounds; budget 3–9 months total.

## 5. What I can do vs. cannot

- **Can:** revise the manuscript, add sections, build and run the answer-quality
  eval, generate figures, tighten novelty framing, format for a specific venue's
  template, and draft the cover letter.
- **Cannot:** submit on your behalf, create accounts, sign the copyright form, pay
  an APC, or guarantee acceptance. Those are yours.

Tell me if you want the answer-quality evaluation built next — that is the step
that most changes the accept/reject odds.
