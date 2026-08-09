Dear Editors,

I am pleased to submit my manuscript, "Dissecting Sequence, Structure, and Data Recency Effects in Enzyme Commission Prediction under Temporal Distribution Shift", for consideration as a Research article in BMC Bioinformatics.

This work studies enzyme function prediction under temporal distribution shift. Rather than presenting Contact-EC as a new leaderboard claim, the manuscript decomposes the contributions of sequence representations, contact-map structure, data recency, label-vocabulary coverage, and fine-tuning strategy. On Swiss-Prot 2023-01 temporal testing, sequence--structure fusion improves over ESM-2-only and contact-only baselines, while a label-shift audit shows that most temporal proteins contain partial or novel EC labels outside the closed-set vocabulary. Perturbation controls further indicate that the contact-map branch uses structural information beyond simple map-density cues. A larger SP-2024 evaluation (N=1,226) reveals a reversal under our mapped evaluation: HIT-EC decreases by -38.9 pp over the longer horizon while Contact-EC improves, so Contact-EC overtakes HIT-EC by +22.4 pp at this horizon even though HIT-EC remains stronger on the one-year SP-2023-01 test. A vocabulary-stratified audit and a Foldseek/TM-score-disjoint split indicate that this decrease is not explained by label-vocabulary mismatch alone.

The manuscript is relevant to BMC Bioinformatics because it addresses benchmark reliability and reproducibility in machine-learning-based enzyme annotation, a central problem in sequence analysis and structural bioinformatics. Code and reproducibility artifacts are available at https://github.com/jamesksh0130/contact-ec.

I confirm that this manuscript is original and is not under consideration elsewhere. AI tools were used only for grammar checking, LaTeX formatting assistance, and code refactoring; all scientific content, experimental design, analysis, and interpretation were performed by the author.

Sincerely,

Seunghyon Kim
School of Applied AI and Entrepreneurship, Handong Global University
Pohang, 37554, Republic of Korea
jamesksh0130@gmail.com
ORCID: 0009-0002-0708-748X
