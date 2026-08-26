#!/usr/bin/env python3
"""Build the MultiHopRAG primary domain: golden set + knowledge base (issue #15).

Vendors a 100- or 120-row golden set from MultiHopRAG (HF ``yixuantt/MultiHopRAG``,
ODC-BY 1.0 — see the ATTRIBUTION.md this script emits) and the knowledge-base
documents its rows cite, staged into ``--out`` (never ``data/`` directly; the
swap into ``data/`` is a deliberate, reviewed step).

Deterministic and reproducible: the HF revision is pinned, every sample draw
comes from one seeded RNG, and ``--check`` re-derives the outputs and
byte-diffs them against the committed ``data/``.

Bucket sourcing:
  * ``multi_hop``    — comparison / inference / temporal queries (stratified)
  * ``distractor``   — answerable queries whose evidence docs have same-category
                       confusables deliberately retained in the KB
  * ``out_of_scope`` — ``null_query`` rows (zero evidence docs; genuinely
                       unanswerable) with ``should_refuse: true``
  * ``direct``       — AUTHORED single-doc questions (MultiHopRAG contains no
                       single-document queries). These 24 rows are the only
                       non-vendored content; review them before any baseline
                       is regenerated against this golden set.

Partition exactness is load-bearing: ``eval.runner._select_mode_examples``
scales mode buckets onto the dev partition and ``select_subset`` raises when a
bucket is short, so ``example_id``s are assigned to make
``partition_dataset`` land the exact per-bucket train/dev/test quotas.

Usage:
    python scripts/build_multihop_domain.py --rows 120 --out build/multihop
    python scripts/build_multihop_domain.py --check            # vs committed data/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anvil.data.golden_set import load_golden_set, select_subset
from anvil.eval.runner import partition_dataset
from anvil.runtime.models import SplitConfig

HF_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
HF_BASE = f"https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/{HF_REVISION}"
RAW_FILES = ("MultiHopRAG.json", "corpus.json")

BUCKETS = ("direct", "multi_hop", "distractor", "out_of_scope")
ID_PREFIX = {
    "direct": "mhr_d",
    "multi_hop": "mhr_m",
    "distractor": "mhr_x",
    "out_of_scope": "mhr_o",
}
KB_DOC_CAP = 350
CONFUSABLES_PER_DISTRACTOR = 5
SHORT_ANSWER_MAX = 60  # answers longer than this make poor must_include strings

REFUSAL_REFERENCE_ANSWER = (
    "The articles in my knowledge base do not cover this. I can only answer "
    "questions the knowledge base supports, and it does not contain the "
    "reporting this question asks about."
)

ATTRIBUTION_MD = """\
# MultiHopRAG

The golden set and knowledge base in this directory are derived from
**MultiHop-RAG** (`yixuantt/MultiHopRAG` on Hugging Face), pinned at revision
`71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`.

- **Paper:** Tang, Yixuan and Yang, Yi. "MultiHop-RAG: Benchmarking
  Retrieval-Augmented Generation for Multi-Hop Queries." arXiv:2401.15391
  (2024). https://arxiv.org/abs/2401.15391
- **Source:** https://huggingface.co/datasets/yixuantt/MultiHopRAG
- **License:** Open Data Commons Attribution License (ODC-BY 1.0),
  https://opendatacommons.org/licenses/by/1-0/

## Derivation

Rows are sampled from the 2,556-query dataset by `scripts/build_multihop_domain.py`
(seed-pinned, reproducible via `--check`): `multi_hop` from comparison /
inference / temporal queries, `out_of_scope` from `null_query` rows,
`distractor` from queries with same-category confusables retained in the KB.
The `direct` bucket is authored (MultiHopRAG contains no single-document
queries) and marked as such in each row's `notes_for_judge`. The knowledge
base is the subset of the 609-document corpus cited by the sampled rows plus
distractor confusables and seeded filler. Document bodies are unmodified.
"""


@dataclass(frozen=True)
class ModeSpec:
    rows: int
    buckets: dict[str, int]


@dataclass(frozen=True)
class VariantSpec:
    buckets: dict[str, int]  # golden-set rows per bucket
    quotas: dict[str, tuple[int, int, int]]  # per bucket: (train, dev, test)
    train_ratio: float
    dev_ratio: float
    modes: dict[str, ModeSpec]


# Mode bucket counts are declared at FULL-SET scale: with the split enabled,
# `_select_mode_examples` multiplies by dev_ratio (round, min 1) and selects
# from the dev partition. `rows` documents the resulting dev count for
# non-test modes; only test mode reads `rows`.
VARIANTS: dict[int, VariantSpec] = {
    100: VariantSpec(
        buckets={"direct": 20, "multi_hop": 40, "distractor": 20, "out_of_scope": 20},
        quotas={
            "direct": (8, 8, 4),
            "multi_hop": (16, 16, 8),
            "distractor": (8, 8, 4),
            "out_of_scope": (8, 8, 4),
        },
        train_ratio=0.4,
        dev_ratio=0.4,
        modes={
            "quick": ModeSpec(8, {"direct": 5, "multi_hop": 5, "distractor": 5, "out_of_scope": 5}),
            "standard": ModeSpec(
                12, {"direct": 8, "multi_hop": 12, "distractor": 5, "out_of_scope": 5}
            ),
            "full": ModeSpec(
                40, {"direct": 20, "multi_hop": 40, "distractor": 20, "out_of_scope": 20}
            ),
            "test": ModeSpec(20, {"direct": 4, "multi_hop": 8, "distractor": 4, "out_of_scope": 4}),
        },
    ),
    120: VariantSpec(
        buckets={"direct": 24, "multi_hop": 48, "distractor": 24, "out_of_scope": 24},
        quotas={
            "direct": (8, 10, 6),
            "multi_hop": (16, 20, 12),
            "distractor": (8, 10, 6),
            "out_of_scope": (8, 10, 6),
        },
        train_ratio=0.3333,
        dev_ratio=0.4167,
        modes={
            "quick": ModeSpec(9, {"direct": 5, "multi_hop": 7, "distractor": 5, "out_of_scope": 5}),
            "standard": ModeSpec(
                15, {"direct": 7, "multi_hop": 14, "distractor": 6, "out_of_scope": 6}
            ),
            "full": ModeSpec(
                50, {"direct": 24, "multi_hop": 48, "distractor": 24, "out_of_scope": 24}
            ),
            "test": ModeSpec(
                30, {"direct": 6, "multi_hop": 12, "distractor": 6, "out_of_scope": 6}
            ),
        },
    ),
}

# ---------------------------------------------------------------------------
# Authored direct rows (MultiHopRAG has no single-document queries).
#
# Each row is answerable from exactly one corpus document; the reference
# answer paraphrases that document's extracted evidence fact(s). Traps in
# `must_not_include` are strings from other documents in the KB pool — a
# response containing them absorbed the wrong article. FORCE_INCLUDE_URLS
# below guarantees the trap-bearing documents land in the KB.
# ---------------------------------------------------------------------------
DIRECT_ROWS: list[dict[str, Any]] = [
    {
        "url": "https://fortune.com/crypto/2023/10/04/sam-bankman-fried-lawyers-opening-statements"
        "-witnesses-marc-antoine-julliard-adam-yedidia-ftx-commodities-trader-developer/",
        "query": "What did each side argue in opening statements at Sam Bankman-Fried's trial?",
        "reference_answer": "Prosecutor Rehn declared to the jury that everything "
        "Bankman-Fried built 'was built on lies.' The defense countered that he was a "
        "good 'boy' of crypto, not a bad 'man' — a well-intentioned founder, not a "
        "fraudster.",
        "must_include": ["built on lies", "Rehn"],
        "must_not_include": ["seven counts", "Warren Buffet"],
        "notes_for_judge": "Authored direct row. Single-doc answer from the Fortune "
        "opening-statements report. The 'built on lies' quote is the prosecution's "
        "framing; charge counts and pre-trial reputation belong to other coverage.",
    },
    {
        "url": "https://www.musicbusinessworldwide.com/sony-musics-artists-arent-involved-in"
        "-youtubes-new-cloned-voice-ai-experiment-not-unrelated-googles-recent-filing"
        "-with-the-us-copyright-office/",
        "query": "Which major music company is not taking part in YouTube's Dream Track "
        "voice-cloning experiment?",
        "reference_answer": "Sony Music's artists are not involved in Dream Track, "
        "YouTube's experiment letting creators clone stars' vocals via AI with official "
        "consent. Sources close to Sony stressed its broader relationship with YouTube "
        "and YouTube Music remains harmonious.",
        "must_include": ["Dream Track", "Sony Music"],
        "must_not_include": ["Pixel 8", "ad-free tier"],
        "notes_for_judge": "Authored direct row. Only the MBW doc answers this. UMG and "
        "Warner appear in the doc as comparisons, not as the answer; do not name Sony "
        "artists as participants.",
    },
    {
        "url": "https://fortune.com/2023/11/18/how-did-openai-fire-sam-altman-greg-brockman"
        "-rogue-board/",
        "query": "How did Greg Brockman respond to Sam Altman's firing from OpenAI?",
        "reference_answer": "Brockman, OpenAI's chairman, quit in protest and accused the "
        "board of going rogue, saying 'Sam and I are shocked and saddened by what the "
        "board did.' Gartner analyst Arun Chandrasekaran called Altman's exit shocking, "
        "noting he had been the face of generative AI.",
        "must_include": ["Brockman", "shocked and saddened"],
        "must_not_include": ["Rehn", "white horse"],
        "notes_for_judge": "Authored direct row. Brockman's quote is first-person plural. "
        "Do not conflate with Sam Bankman-Fried trial coverage.",
    },
    {
        "url": "https://www.smh.com.au/business/markets/asx-set-to-open-lower-as-wall-street"
        "-closes-september-with-more-losses-20231002-p5e8zn.html"
        "?ref=rss&utm_medium=rss&utm_source=rss_business",
        "query": "Why could postponed US government data reports complicate the Fed's "
        "rate decisions, per the SMH markets wrap?",
        "reference_answer": "The Fed has insisted it will base upcoming interest-rate "
        "decisions on what incoming data say about the economy, so postponements of key "
        "reports could complicate those decisions — a worry as Wall Street closed "
        "September with more losses.",
        "must_include": ["Fed", "postponements"],
        "must_not_include": ["$A rises", "keep its overnight rate higher"],
        "notes_for_judge": "Authored direct row. The SMH and The Age ASX wraps are "
        "near-identical in topic; only the SMH one discusses data postponements. The "
        "Age's 'highest since 2001' framing is the distractor.",
    },
    {
        "url": "https://www.cbssports.com/fantasy/football/news/nfl-fantasy-football-week-5"
        "-lineup-decisions-starts-sits-sleepers-busts-to-know-for-every-game/",
        "query": "How did Kirk Cousins perform in Week 4, according to the Week 5 fantasy "
        "lineup report?",
        "reference_answer": "Cousins threw only 19 times in a win, with two touchdowns "
        "but just 13 Fantasy points — and Week 4 was the first time all year the Vikings "
        "did not throw on at least 69% of their snaps.",
        "must_include": ["19 times", "13 Fantasy points"],
        "must_not_include": ["Gaston Edul", "J.J. McCarthy"],
        "notes_for_judge": "Authored direct row. Week 5 report only; the Week 6 report's "
        "Purdy pressure stats and Achane injury are same-source distractors.",
    },
    {
        "url": "https://www.sportingnews.com/us/soccer/news/inter-miami-vs-fc-cincinnati-live"
        "-score-result-highlights-mls/6288f57a14a413f02512e266",
        "query": "What was reported about Lionel Messi's availability for Inter Miami's "
        "playoff match against FC Cincinnati?",
        "reference_answer": "Argentine journalist Gaston Edul reported Inter Miami were "
        "likely to have Lionel Messi back for the match in some capacity. Cincinnati "
        "arrived as Supporters' Shield winners, confirmed as regular-season champions on "
        "the Wednesday despite defeat.",
        "must_include": ["Gaston Edul", "Messi", "Supporters' Shield"],
        "must_not_include": ["J.J. McCarthy", "Ric Flair"],
        "notes_for_judge": "Authored direct row. Live-blog doc; the Edul report is the "
        "availability claim. Michigan sign-stealing content is a same-source distractor.",
    },
    {
        "url": "https://www.sportingnews.com/us/ncaa-football/news/michigan-scandal-winners"
        "-losers-jim-harbaugh-tony-petitti/52ad1c9ec667b556323fa41c",
        "query": "Who does the Michigan sign-stealing scandal report name as its central "
        "winners and losers?",
        "reference_answer": "The report runs from Jim Harbaugh to Tony Petitti, with "
        "Michigan — led by J.J. McCarthy, Blake Corum and a defense allowing 7.5 points "
        "per game — thrust into a villain role, complete with visits from Ric Flair.",
        "must_include": ["Jim Harbaugh", "Tony Petitti", "7.5 points"],
        "must_not_include": ["Messi", "Supporters' Shield"],
        "notes_for_judge": "Authored direct row. Names beyond Harbaugh, Petitti, "
        "McCarthy, Corum and Flair are not established by this doc.",
    },
    {
        "url": "https://www.sportingnews.com/us/nfl/news/bears-vikings-live-score-highlights"
        "-monday-night-football/9e0ddaf702f99e7aec9645db",
        "query": "How did the Bears' defense perform against the Vikings on Monday Night Football?",
        "reference_answer": "The Bears' defense was all over the Vikings, and Minnesota "
        "could not get out of its own way.",
        "must_include": ["all over the Vikings"],
        "must_not_include": ["13 Fantasy points", "De'Von Achane"],
        "notes_for_judge": "Authored direct row. Live-game blog; defensive dominance is "
        "what the early report establishes. Season-level Cousins stats belong to fantasy "
        "coverage, not this game.",
    },
    {
        "url": "https://techcrunch.com/2023/09/28/chatgpt-everything-to-know-about-the-ai-chatbot/",
        "query": "What did OpenAI announce about GPT-4's vision capabilities?",
        "reference_answer": "OpenAI announced that GPT-4 with vision would become "
        "available alongside the launch of the GPT-4 Turbo API.",
        "must_include": ["GPT-4 with vision", "GPT-4 Turbo"],
        "must_not_include": ["seven counts", "Eddy Cue"],
        "notes_for_judge": "Authored direct row. Only the vision / Turbo-API timing is "
        "established by this explainer.",
    },
    {
        "url": "https://techcrunch.com/2023/10/01/ftx-lawsuit-timeline/",
        "query": "What charges does Sam Bankman-Fried's criminal trial decide, and how was "
        "he regarded before FTX's collapse?",
        "reference_answer": "The trial determines whether the former FTX CEO is guilty of "
        "seven counts of fraud and conspiracy. Before his fall, Bankman-Fried was "
        "compared to Warren Buffett and called the white horse of crypto.",
        "must_include": ["seven counts", "white horse"],
        "must_not_include": ["built on lies", "Rehn"],
        "notes_for_judge": "Authored direct row. Timeline doc. The 'built on lies' quote "
        "belongs to trial coverage, not this retrospective.",
    },
    {
        "url": "https://www.theverge.com/2023/9/26/23891037/apple-eddy-cue-testimony-us-google",
        "query": "Why is the Justice Department scrutinizing Apple's deal with Google, and "
        "what is Apple's defense?",
        "reference_answer": "The Justice Department is focused on the deals Google makes "
        "— with Apple, Samsung, Mozilla and others — to remain the default search engine "
        "on practically every platform. Apple, through Eddy Cue's testimony, defended "
        "the deal, arguing 'there wasn't a valid alternative.'",
        "must_include": ["default search engine", "valid alternative"],
        "must_not_include": ["Pixel 8", "ad-free tier"],
        "notes_for_judge": "Authored direct row. Cue's quote is the defense's core claim; "
        "Samsung and Mozilla appear as other Google partners, not defendants.",
    },
    {
        "url": "https://techcrunch.com/2023/10/13/uber-sexual-assault-survivors-call-for-in"
        "-car-cameras-tech-upgrades/",
        "query": "What do the lawsuits against Uber claim, and what remedies are survivors "
        "calling for?",
        "reference_answer": "Hundreds of women have filed lawsuits claiming Uber has not "
        "done enough to prevent sexual assault by drivers; survivors are calling for "
        "in-car cameras and technology upgrades.",
        "must_include": ["in-car cameras", "Hundreds of women"],
        "must_not_include": ["GPT-4", "Eddy Cue"],
        "notes_for_judge": "Authored direct row. Do not generalize beyond the lawsuits' "
        "claims as reported.",
    },
    {
        "url": "https://www.theage.com.au/culture/music/when-pop-culture-and-sport-collide-a"
        "-timeline-of-taylor-swift-s-nfl-takeover-20230926-p5e7lu.html"
        "?ref=rss&utm_medium=rss&utm_source=rss_culture",
        "query": "What did Travis Kelce reveal about his first attempt to give Taylor "
        "Swift his number?",
        "reference_answer": "On New Heights, the podcast he shares with his brother, "
        "Kelce said he intended to give Swift a friendship bracelet with his number on it "
        "during her Eras Tour concert in Kansas City.",
        "must_include": ["friendship bracelet", "New Heights"],
        "must_not_include": ["Anti-Hero", "Jaden"],
        "notes_for_judge": "Authored direct row. The Age's timeline doc. The later "
        "Independent 'secret start' interview is the distractor.",
    },
    {
        "url": "https://www.independent.co.uk/life-style/jada-pinkett-will-smith-prenup"
        "-b2430529.html",
        "query": "What has Jada Pinkett Smith said about why she and Will Smith never "
        "signed a prenup, and about their children?",
        "reference_answer": "She has spoken repeatedly about their long-term commitment "
        "as the reason no prenup was needed, and praised their children as 'little "
        "gurus' who helped her grow.",
        "must_include": ["prenup", "little gurus"],
        "must_not_include": ["separated since 2016", "Alsina"],
        "notes_for_judge": "Authored direct row. Prenup interview doc only. The "
        "separation-timeline doc (Alsina, 2016) is the same-couple distractor.",
    },
    {
        "url": "https://www.independent.co.uk/life-style/justin-timberlake-sam-asghari"
        "-britney-spears-b2431507.html",
        "query": "According to the Britney Spears relationship timeline, what rumour "
        "followed the Justin Timberlake music video?",
        "reference_answer": "The music video — which featured a woman with blonde hair — "
        "sparked rumours that Spears and Timberlake broke up because she allegedly "
        "cheated on him.",
        "must_include": ["blonde", "cheated"],
        "must_not_include": ["Anti-Hero", "paparazzi"],
        "notes_for_judge": "Authored direct row. The video rumour belongs to the "
        "Timberlake era, not the Asghari marriage; confusing the two is the distractor "
        "failure.",
    },
    {
        "url": "https://www.independent.co.uk/life-style/taylor-swift-travis-kelce"
        "-relationship-b2459441.html",
        "query": "How does Taylor Swift describe her approach to paparazzi attention in "
        "the Time interview?",
        "reference_answer": "The 'Anti-Hero' singer told Time she tries not to let the "
        "paparazzi get to her, even though so many of her outings with friends make "
        "headlines.",
        "must_include": ["Anti-Hero", "paparazzi"],
        "must_not_include": ["Jaden", "little gurus"],
        "notes_for_judge": "Authored direct row. The Time interview doc. The "
        "friendship-bracelet story is from The Age's timeline — the classic cross-doc "
        "conflation.",
    },
    {
        "url": "https://www.snexplores.org/article/quantum-dots-technology-2023-nobel-prize"
        "-chemistry",
        "query": "What makes quantum dots promising for applications such as solar panels?",
        "reference_answer": "Quantum dots can contain the same molecules yet have "
        "different colors and qualities depending on their size, so they could be used "
        "to build solar panels that soak up sunlight well in different conditions — work "
        "that won the 2023 chemistry Nobel.",
        "must_include": ["size", "solar panels"],
        "must_not_include": ["zinc", "hydrogels"],
        "notes_for_judge": "Authored direct row. Single-doc. The zinc-iodine battery "
        "research is a same-category distractor.",
    },
    {
        "url": "https://news.yahoo.com/scientist-reckons-climate-grief-130602750.html",
        "query": "What lifestyle changes did the climate scientist in the Yahoo profile "
        "make, and by how much did he cut his emissions?",
        "reference_answer": "He stopped flying, became a vegetarian and ditched "
        "gasoline-powered cars — he drives a Tesla — cutting his personal emissions by "
        "about 90%, according to his own math.",
        "must_include": ["vegetarian", "Tesla", "90%"],
        "must_not_include": ["quantum dots", "Nobel"],
        "notes_for_judge": "Authored direct row. The 90% figure is the scientist's own "
        "estimate; attribute it as such.",
    },
    {
        "url": "https://www.livescience.com/health/hiv/we-could-end-the-aids-epidemic-in-less"
        "-than-a-decade-heres-how",
        "query": "What do patients on antiretroviral therapy need, and what do experts say "
        "must continue, per the AIDS epidemic report?",
        "reference_answer": "Regardless of the type of ART they take, patients should "
        "have their viral load checked regularly, and experts stress continued "
        "investment to find a vaccine and a cure — steps that could end the AIDS "
        "epidemic in less than a decade.",
        "must_include": ["viral load", "vaccine"],
        "must_not_include": ["Ozempic", "stomach paralysis"],
        "notes_for_judge": "Authored direct row. The 'less than a decade' framing is "
        "expert aspiration, not a finding.",
    },
    {
        "url": "https://www.advancedsciencenews.com/bringing-aqueous-rechargeable-zinc-iodine"
        "-batteries-to-the-mainstream-energy-market/",
        "query": "What is the goal of the new research into aqueous zinc-iodine batteries?",
        "reference_answer": "The research aims to improve the stability and safety of "
        "alternatives to rechargeable lithium-ion batteries, using aqueous zinc and "
        "hydrogels to bring zinc-iodine batteries toward the mainstream energy market.",
        "must_include": ["stability", "hydrogels"],
        "must_not_include": ["quantum dots", "solar panels"],
        "notes_for_judge": "Authored direct row. Do not overclaim commercial readiness; "
        "the work is research-stage.",
    },
    {
        "url": "https://www.foxnews.com/health/cell-phone-shocker-97-percent-kids-use-device"
        "-during-school-hours-beyond-study",
        "query": "What did the study cited by Fox News find about children's phone use and TikTok?",
        "reference_answer": "The study found 97% of kids use their device during school "
        "hours and beyond, with TikTok used by half of participants for nearly two hours "
        "per day on average.",
        "must_include": ["97%", "TikTok", "two hours"],
        "must_not_include": ["Zoom fatigue", "viral load"],
        "notes_for_judge": "Authored direct row. A single study as reported; do not merge "
        "with the remote-work 'Zoom fatigue' piece.",
    },
    {
        "url": "https://www.foxnews.com/health/fasting-could-reduce-signs-alzheimers-disease"
        "-studies-suggest-profound-effects",
        "query": "What benefits of fasting did the studies cited by Fox News suggest?",
        "reference_answer": "Studies suggested fasting could reduce signs of Alzheimer's "
        "disease, with researchers noting 'profound effects'; time-restricted eating can "
        "also improve sleep quality, helping the brain recover better.",
        "must_include": ["profound effects", "time-restricted"],
        "must_not_include": ["viral load", "ART"],
        "notes_for_judge": "Authored direct row. 'Suggested' language matters — these are "
        "study suggestions, not clinical recommendations.",
    },
    {
        "url": "https://www.foxnews.com/health/losing-leg-flu-virginia-woman-urges-people-get"
        "-vaccinated-dont-waste-time",
        "query": "Why is the Virginia woman who lost her leg to the flu urging people to "
        "get vaccinated?",
        "reference_answer": "After losing her leg to the flu, she urges people not to "
        "waste time getting vaccinated, warning that a viral illness like influenza can "
        "set you up for something more serious.",
        "must_include": ["vaccinated", "influenza"],
        "must_not_include": ["Ozempic", "Wegovy"],
        "notes_for_judge": "Authored direct row. Human-interest report; the 'set you up "
        "for something more serious' warning is the medical through-line.",
    },
    {
        "url": "https://www.foxnews.com/health/ozempic-wegovy-may-be-linked-stomach-paralysis"
        "-other-digestive-issues-large-scale-study",
        "query": "What did the large-scale study link Ozempic and Wegovy to, and how common "
        "were the complications?",
        "reference_answer": "The study linked Ozempic and Wegovy to stomach paralysis and "
        "other digestive issues. The complications were rare, though researchers found "
        "them concerning given the millions using these medications worldwide.",
        "must_include": ["stomach paralysis", "rare"],
        "must_not_include": ["fasting", "viral load"],
        "notes_for_judge": "Authored direct row. The rarity of the complications is as "
        "much the story as the link — omitting it misleads.",
    },
]

# Documents that must land in the KB because authored-row traps quote them.
FORCE_INCLUDE_URLS = {
    "https://www.cbssports.com/fantasy/football/news/nfl-fantasy-football-week-6-lineup"
    "-decisions-starts-sits-sleepers-busts-to-know-for-every-game/",
    "https://www.independent.co.uk/life-style/will-smith-jada-separation-divorce-b2429576.html",
    "https://www.foxnews.com/health/zoom-fatigue-common-struggle-remote-workers-heres-how"
    "-handle-according-experts",
    "https://techcrunch.com/2023/10/07/sam-altman-backs-a-teens-startup-google-unveils-the"
    "-pixel-8-and-tiktok-tests-an-ad-free-tier/",
    "https://www.theage.com.au/business/markets/asx-set-to-edge-up-as-wall-street-steadies"
    "-a-rises-20231005-p5e9va.html?ref=rss&utm_medium=rss&utm_source=rss_business",
}

_TOPIC_STOPWORDS = {
    "The",
    "And",
    "But",
    "For",
    "With",
    "What",
    "Who",
    "How",
    "Why",
    "When",
    "Where",
    "Which",
    "Considering",
    "Given",
    "Based",
    "According",
    "Between",
    "Among",
    "After",
    "Before",
    "During",
    "While",
    "News",
    "Report",
    "Article",
    "BBC",
    "Bloomberg",
    "CNN",
    "Reuters",
    "Forbes",
    "Guardian",
    "Verge",
    "TechCrunch",
    "Mashable",
    "CNBC",
    "Fox",
}
_YESNO_ANSWER = re.compile(r"^(yes|no|agree|disagree)\.?$", re.IGNORECASE)
_TOPIC_RE = re.compile(r"\b[A-Z][\w'&.-]*(?:\s+[A-Z][\w'&.-]*)*")


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def fetch_raw(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        dest = out_dir / name
        if dest.exists():
            continue
        print(f"fetching {name} @ {HF_REVISION[:12]}…")
        with urllib.request.urlopen(f"{HF_BASE}/{name}", timeout=120) as resp:
            dest.write_bytes(resp.read())


def load_raw(raw_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries: list[dict[str, Any]] = json.loads((raw_dir / "MultiHopRAG.json").read_text())
    corpus: list[dict[str, Any]] = json.loads((raw_dir / "corpus.json").read_text())
    return queries, corpus


def _partition_of(example_id: str, *, seed: int, train_cut: float, dev_cut: float) -> str:
    """Replica of eval.runner.partition_dataset's assignment for a single id."""
    digest = hashlib.md5(  # noqa: S324 - deterministic partitioning, not security
        f"{seed}:{example_id}".encode(), usedforsecurity=False
    ).hexdigest()
    fraction = int(digest, 16) / (2**128)
    if fraction < train_cut:
        return "train"
    if fraction < dev_cut:
        return "dev"
    return "test"


def assign_ids(
    bucket: str,
    rows: list[dict[str, Any]],
    quota: tuple[int, int, int],
    spec: VariantSpec,
    seed: int,
) -> list[dict[str, Any]]:
    """Number rows so partition_dataset lands the exact (train, dev, test) quota."""
    targets = {"train": quota[0], "dev": quota[1], "test": quota[2]}
    counts = {"train": 0, "dev": 0, "test": 0}
    out: list[dict[str, Any]] = []
    counter = 0
    for row in rows:
        while True:
            counter += 1
            eid = f"{ID_PREFIX[bucket]}_{counter:03d}"
            part = _partition_of(
                eid,
                seed=seed,
                train_cut=spec.train_ratio,
                dev_cut=spec.train_ratio + spec.dev_ratio,
            )
            if counts[part] < targets[part]:
                counts[part] += 1
                out.append({**row, "example_id": eid})
                break
    if counts != targets:
        _fail(f"{bucket}: id pinning missed quota {targets}, got {counts}")
    return out


def pick_topic(query: str) -> str | None:
    """Longest capitalized phrase in the query — a refusal should name its topic."""
    best: str | None = None
    for match in _TOPIC_RE.finditer(query):
        phrase = match.group(0).strip()
        head = phrase.split()[0]
        if head in _TOPIC_STOPWORDS and len(phrase.split()) == 1:
            continue
        if head in _TOPIC_STOPWORDS:
            phrase = " ".join(phrase.split()[1:])
            if not phrase:
                continue
        if best is None or len(phrase) > len(best):
            best = phrase
    return best


def _yaml_str(value: str) -> str:
    """JSON double-quoted scalar — valid YAML for arbitrary title/source text."""
    return json.dumps(value, ensure_ascii=False)


def render_doc(doc_id: str, raw: dict[str, Any]) -> str:
    lines = ["---", f"doc_id: {doc_id}", f"title: {_yaml_str(raw['title'])}"]
    lines.append(f"category: {_yaml_str(raw['category'])}")
    lines.append(f"source: {_yaml_str(raw['source'])}")
    lines.append(f"url: {raw['url']}")
    lines.append(f"published_at: {raw['published_at']}")
    if raw.get("author"):
        lines.append(f"author: {_yaml_str(raw['author'])}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {raw['title']}")
    lines.append("")
    lines.append(raw["body"].rstrip())
    lines.append("")
    return "\n".join(lines)


def _doc_text(raw: dict[str, Any]) -> str:
    return f"{raw['title']}\n{raw['body']}"


def build(
    spec: VariantSpec, *, seed: int, queries: list[dict[str, Any]], corpus: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Returns (golden_rows, docs_by_id)."""
    rng = random.Random(seed)
    by_url = {d["url"]: d for d in corpus}
    if len(by_url) != len(corpus):
        _fail("corpus urls are not unique")

    for q in queries:
        for e in q["evidence_list"]:
            if e["url"] not in by_url:
                _fail(f"evidence url missing from corpus: {e['url']}")

    answerable = [q for q in queries if q["question_type"] != "null_query"]
    null_rows = [q for q in queries if q["question_type"] == "null_query"]

    def short_enough(q: dict[str, Any]) -> bool:
        # yes/no answers give degenerate must_include strings and leave the
        # correctness judge a one-word reference to grade against
        return len(q["answer"]) <= SHORT_ANSWER_MAX and not _YESNO_ANSWER.match(q["answer"].strip())

    # --- distractor: answerable rows with same-category confusables available
    distractor_pool = [q for q in answerable if short_enough(q)]
    rng.shuffle(distractor_pool)
    distractor_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    used_queries: set[int] = set()
    for q in distractor_pool:
        if len(distractor_rows) >= spec.buckets["distractor"]:
            break
        cats = {by_url[e["url"]]["category"] for e in q["evidence_list"]}
        cited_urls = {e["url"] for e in q["evidence_list"]}
        confusables = [d for d in corpus if d["category"] in cats and d["url"] not in cited_urls]
        if len(confusables) < 2:
            continue
        chosen = rng.sample(confusables, k=min(CONFUSABLES_PER_DISTRACTOR, len(confusables)))
        distractor_rows.append((q, chosen))
        used_queries.add(id(q))

    # --- multi_hop: stratified over the three answerable types
    n_mh = spec.buckets["multi_hop"]
    strata = [n_mh // 3 + (1 if i < n_mh % 3 else 0) for i in range(3)]
    multi_rows: list[dict[str, Any]] = []
    for qtype, want in zip(
        ("comparison_query", "inference_query", "temporal_query"), strata, strict=True
    ):
        pool = [
            q
            for q in answerable
            if q["question_type"] == qtype and short_enough(q) and id(q) not in used_queries
        ]
        rng.shuffle(pool)
        take = pool[:want]
        if len(take) < want:
            _fail(f"multi_hop: only {len(take)} eligible {qtype} rows")
        multi_rows.extend(take)
        used_queries.update(id(q) for q in take)

    # --- out_of_scope: null_query rows
    oos_pool = list(null_rows)
    rng.shuffle(oos_pool)
    oos_rows = oos_pool[: spec.buckets["out_of_scope"]]
    if len(oos_rows) < spec.buckets["out_of_scope"]:
        _fail(f"out_of_scope: only {len(oos_rows)} null rows available")

    # --- direct: authored table
    direct_specs = DIRECT_ROWS[: spec.buckets["direct"]]
    for entry in direct_specs:
        if entry["url"] not in by_url:
            _fail(f"authored direct row cites unknown url: {entry['url']}")

    # --- KB pool: evidence docs + confusables + forced + filler
    kb_urls: set[str] = set()
    for q in multi_rows:
        kb_urls.update(e["url"] for e in q["evidence_list"])
    for q, conf in distractor_rows:
        kb_urls.update(e["url"] for e in q["evidence_list"])
        kb_urls.update(d["url"] for d in conf)
    kb_urls.update(e["url"] for e in direct_specs)
    kb_urls.update(FORCE_INCLUDE_URLS)
    remainder = sorted(u for u in by_url if u not in kb_urls)
    filler = rng.sample(remainder, k=max(0, min(KB_DOC_CAP - len(kb_urls), len(remainder))))
    kb_urls.update(filler)

    doc_ids = {url: f"doc_{i:03d}" for i, url in enumerate(sorted(kb_urls), start=1)}
    docs_by_id = {doc_ids[u]: by_url[u] for u in sorted(kb_urls)}

    # --- wrong-answer entity pool for must_not_include fallbacks
    sampled_answers = sorted(
        {q["answer"] for q in multi_rows} | {q["answer"] for q, _ in distractor_rows}
    )
    body_cache: dict[str, str] = {}

    def doc_text(doc_id: str) -> str:
        if doc_id not in body_cache:
            body_cache[doc_id] = _doc_text(docs_by_id[doc_id])
        return body_cache[doc_id]

    def trap_strings(cited_ids: list[str], answer: str, n: int = 2) -> list[str]:
        """Up to n strings present in some non-cited KB doc, absent from all cited docs."""
        cited_text = "\n".join(doc_text(d) for d in cited_ids)
        traps: list[str] = []
        for cand in sampled_answers:
            distinctive = " " in cand.strip() or any(ch.isdigit() for ch in cand)
            if cand == answer or len(cand) < 4 or not distinctive or cand in cited_text:
                continue
            if any(cand in doc_text(d) for d in docs_by_id if d not in cited_ids):
                traps.append(cand)
            if len(traps) >= n:
                return traps
        for d in sorted(docs_by_id):
            if d in cited_ids:
                continue
            title = docs_by_id[d]["title"]
            if title not in cited_text:
                traps.append(title)
            if len(traps) >= n:
                return traps
        return traps

    # --- golden rows
    golden: dict[str, list[dict[str, Any]]] = {b: [] for b in BUCKETS}

    for entry in direct_specs:
        cited = [doc_ids[entry["url"]]]
        golden["direct"].append(
            {
                "query": entry["query"],
                "category": "direct",
                "expected_doc_ids": cited,
                "reference_answer": entry["reference_answer"],
                "should_refuse": False,
                "expected_citations": cited,
                "must_include": entry["must_include"],
                "must_not_include": entry["must_not_include"],
                "notes_for_judge": entry["notes_for_judge"],
            }
        )

    for q in multi_rows:
        cited = [doc_ids[e["url"]] for e in q["evidence_list"]]
        titles = "; ".join(dict.fromkeys(docs_by_id[d]["title"] for d in cited))
        golden["multi_hop"].append(
            {
                "query": q["query"],
                "category": "multi_hop",
                "expected_doc_ids": cited,
                "reference_answer": q["answer"],
                "should_refuse": False,
                "expected_citations": cited,
                "must_include": [q["answer"]],
                "must_not_include": trap_strings(cited, q["answer"]),
                "notes_for_judge": f"Multi-hop ({q['question_type'].removesuffix('_query')}). "
                f"Requires combining facts from {len(cited)} articles: {titles}. "
                "Answering from a single article or naming an entity from an unrelated "
                "article is a hard fail.",
            }
        )

    for q, conf in distractor_rows:
        cited = [doc_ids[e["url"]] for e in q["evidence_list"]]
        conf_ids = [doc_ids[d["url"]] for d in conf]
        conf_titles = [docs_by_id[d]["title"] for d in conf_ids]
        traps: list[str] = []
        for title in conf_titles:
            cited_text = "\n".join(doc_text(d) for d in cited)
            if title not in cited_text and title not in traps:
                traps.append(title)
            if len(traps) >= 3:
                break
        traps.extend(s for s in trap_strings(cited, q["answer"], n=2) if s not in traps)
        golden["distractor"].append(
            {
                "query": q["query"],
                "category": "distractor",
                "expected_doc_ids": cited,
                "reference_answer": q["answer"],
                "should_refuse": False,
                "expected_citations": cited,
                "must_include": [q["answer"]],
                "must_not_include": traps[:4],
                "notes_for_judge": "Distractor: same-category articles surface on "
                f"retrieval ({'; '.join(conf_titles[:3])}). Do not attribute their "
                "claims; the question is answered only by the cited articles.",
            }
        )

    for q in oos_rows:
        topic = pick_topic(q["query"])
        must_include = ["knowledge base"] + ([topic] if topic else [])
        golden["out_of_scope"].append(
            {
                "query": q["query"],
                "category": "out_of_scope",
                "expected_doc_ids": [],
                "reference_answer": REFUSAL_REFERENCE_ANSWER,
                "should_refuse": True,
                "expected_citations": [],
                "must_include": must_include,
                "must_not_include": [],
                "notes_for_judge": "RefusalAppropriateness: genuinely unanswerable from "
                "this corpus (MultiHopRAG null_query — its evidence is absent by "
                "design). Must refuse without citing any article or inventing facts "
                "about the topic. Any article citation is a hard fail.",
            }
        )

    # --- id pinning, then flatten in bucket-block order
    pinned: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        rows = golden[bucket]
        if len(rows) != spec.buckets[bucket]:
            _fail(f"{bucket}: built {len(rows)} rows, spec wants {spec.buckets[bucket]}")
        pinned.extend(assign_ids(bucket, rows, spec.quotas[bucket], spec, seed))

    return pinned, docs_by_id


def check_golden(
    rows: list[dict[str, Any]], docs_by_id: dict[str, dict[str, Any]], spec: VariantSpec, seed: int
) -> None:
    """Hard-fail integrity checks — the same rules test_primary_domain.py pins."""
    problems: list[str] = []

    def doc_text(doc_id: str) -> str:
        return _doc_text(docs_by_id[doc_id])

    for row in rows:
        eid = row["example_id"]
        cited = row["expected_doc_ids"]
        refusal = row["should_refuse"]
        if refusal != (row["category"] == "out_of_scope"):
            problems.append(f"{eid}: should_refuse != (category == out_of_scope)")
        if refusal and (cited or row["expected_citations"]):
            problems.append(f"{eid}: refusal row cites docs")
        if any(d not in docs_by_id for d in cited):
            problems.append(f"{eid}: cites doc missing from KB")
        if any(d not in cited for d in row["expected_citations"]):
            problems.append(f"{eid}: expected_citations not a subset of expected_doc_ids")
        if not row["must_include"]:
            problems.append(f"{eid}: empty must_include")
        cited_text = "\n".join(doc_text(d) for d in cited)
        for s in row["must_include"]:
            if s not in row["reference_answer"] and s not in cited_text and s not in row["query"]:
                problems.append(f"{eid}: must_include {s!r} in neither answer, cited docs, query")
        if not refusal:
            if not row["must_not_include"]:
                problems.append(f"{eid}: empty must_not_include on non-refusal row")
            for s in row["must_not_include"]:
                if s in cited_text:
                    problems.append(f"{eid}: must_not_include {s!r} appears in a cited doc")
                if not any(s in doc_text(d) for d in docs_by_id if d not in cited):
                    problems.append(f"{eid}: must_not_include {s!r} absent from non-cited KB")

    split = SplitConfig(
        enabled=True, train_ratio=spec.train_ratio, dev_ratio=spec.dev_ratio, seed=seed
    )
    train, dev, test = partition_dataset(rows, split)
    table: dict[str, dict[str, int]] = {b: {} for b in BUCKETS}
    for part, examples in (("train", train), ("dev", dev), ("test", test)):
        for bucket in BUCKETS:
            table[bucket][part] = sum(1 for e in examples if e["category"] == bucket)
    for bucket in BUCKETS:
        want = spec.quotas[bucket]
        got = (table[bucket]["train"], table[bucket]["dev"], table[bucket]["test"])
        if got != want:
            problems.append(f"partition {bucket}: want {want}, got {got}")

    for mode, mspec in spec.modes.items():
        if mode == "test":
            if len(test) != mspec.rows:
                problems.append(f"test mode: partition size {len(test)} != rows {mspec.rows}")
            continue
        scaled = {b: max(1, round(c * spec.dev_ratio)) for b, c in mspec.buckets.items()}
        try:
            selected = select_subset(dev, buckets=scaled)
        except ValueError as exc:
            problems.append(f"{mode} mode selection failed: {exc}")
            continue
        if len(selected) != mspec.rows:
            problems.append(f"{mode} mode: {len(selected)} rows selected, spec says {mspec.rows}")

    if problems:
        for p in problems:
            print(f"  FAIL {p}", file=sys.stderr)
        _fail(f"{len(problems)} integrity problem(s)")


def write_domain(
    out_dir: Path, rows: list[dict[str, Any]], docs_by_id: dict[str, dict[str, Any]]
) -> None:
    kb_dir = out_dir / "kb"
    kb_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "golden_set.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for doc_id, raw in docs_by_id.items():
        (kb_dir / f"{doc_id}.md").write_text(render_doc(doc_id, raw), encoding="utf-8")
    (out_dir / "ATTRIBUTION.md").write_text(ATTRIBUTION_MD, encoding="utf-8")


def run_pipeline(args: argparse.Namespace, out_dir: Path) -> None:
    spec = VARIANTS[args.rows]
    raw_dir = Path(args.input_dir) if args.input_dir else out_dir / "raw"
    if not args.input_dir:
        fetch_raw(raw_dir)
    queries, corpus = load_raw(raw_dir)
    rows, docs_by_id = build(spec, seed=args.seed, queries=queries, corpus=corpus)
    check_golden(rows, docs_by_id, spec, args.seed)
    write_domain(out_dir, rows, docs_by_id)
    # validate the written artifact with the real loader
    loaded = load_golden_set(out_dir / "golden_set.jsonl")
    if len(loaded) != sum(spec.buckets.values()):
        _fail(f"loader read {len(loaded)} rows, spec wants {sum(spec.buckets.values())}")

    split = SplitConfig(
        enabled=True, train_ratio=spec.train_ratio, dev_ratio=spec.dev_ratio, seed=args.seed
    )
    train, dev, test = partition_dataset(rows, split)
    print(f"golden set: {len(rows)} rows ({', '.join(f'{b} {spec.buckets[b]}' for b in BUCKETS)})")
    print(f"partition:  train {len(train)} / dev {len(dev)} / test {len(test)}")
    for bucket in BUCKETS:
        q = spec.quotas[bucket]
        print(f"  {bucket:<13} train {q[0]:>2} / dev {q[1]:>2} / test {q[2]:>2}")
    for mode, mspec in spec.modes.items():
        if mode == "test":
            print(f"mode test:     {len(test)} rows (whole test partition)")
        else:
            scaled = {b: max(1, round(c * spec.dev_ratio)) for b, c in mspec.buckets.items()}
            print(f"mode {mode}:".ljust(12), f"{sum(scaled.values())} dev rows  {scaled}")
    print(f"kb: {len(docs_by_id)} docs -> {out_dir / 'kb'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, choices=sorted(VARIANTS), default=120)
    parser.add_argument("--out", type=Path, default=Path("build/multihop"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dir", type=Path, default=None, help="replay from local raw JSON")
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-derive and byte-diff against committed data/ (reproducibility proof)",
    )
    args = parser.parse_args()

    if args.check:
        data_dir = Path("data")
        if not (data_dir / "golden_set.jsonl").is_file():
            _fail("data/golden_set.jsonl not found — run from the repo root")
        with tempfile.TemporaryDirectory() as tmp:
            run_pipeline(args, Path(tmp))
            expected = sorted(p.relative_to(tmp) for p in Path(tmp).rglob("*") if p.is_file())
            problems = 0
            for rel in expected:
                if rel.parts[0] == "raw":
                    continue
                committed = data_dir / rel
                rebuilt = Path(tmp) / rel
                if not committed.is_file():
                    print(f"  MISSING {rel}", file=sys.stderr)
                    problems += 1
                elif committed.read_bytes() != rebuilt.read_bytes():
                    print(f"  DIFFERS {rel}", file=sys.stderr)
                    problems += 1
            extra = [
                p
                for p in data_dir.rglob("*")
                if p.is_file() and p.relative_to(data_dir) not in expected and p.suffix == ".md"
            ]
            for p in extra:
                print(f"  EXTRA {p}", file=sys.stderr)
                problems += 1
            if problems:
                _fail(f"--check failed: {problems} mismatch(es) vs committed data/")
        print("--check OK: committed data/ reproduces byte-for-byte")
        return 0

    run_pipeline(args, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
