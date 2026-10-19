# RAG eval v3: 20 domande appaiate su 5 paper

10 coppie: una domanda con ID arXiv espliciti e la stessa domanda senza ID. Domande in inglese, come v2. Il set v2 resta invariato. Tutte le risposte attese sono bozze da revisionare, basate sui testi locali estratti; non sono gold label validate da un esperto.

- 10 esplicite e 10 implicite. Le implicite mantengono descrittori/modelli, senza ID, URL o titoli di paper.
- 16 rispondibili e 4 insufficient-evidence (2 coppie).
- 3 coppie di confronto richiedono tutte le fonti: 6 domande.
- Le domande esplicite dei due casi senza risposta nominano rispettivamente 5 e 2 paper; non sono ulteriori confronti rispondibili.
- Le keyword di sezione sono vuote per entrambe le varianti: non aggiungiamo suggerimenti derivati dalla risposta attesa al retrieval.
- Nessun nuovo benchmark eseguito.

File operativo: `rag_eval_set_v3_paired.jsonl`. Risposte/evidenze: `rag_eval_set_v3_reference_answers_draft.jsonl`, da non passare come input di generazione. Gli ID `expected_*` servono solo alla valutazione; il filtro legge esclusivamente gli ID presenti nella domanda.

Gli expected-source hit sulle domande esplicite sono favoriti dal filtro. Confrontare soprattutto evidenza sufficiente, correttezza, supporto delle citazioni e astensione. Per le implicite, fonti diverse da quelle canoniche possono essere valide: controllarle manualmente senza dichiarare automaticamente errata la risposta.

## eval_v3_pair_01

**Con fonti esplicite** (`eval_v3_pair_01_explicit`)

Using arXiv:1705.02355v2, what is the name and generative model family of the fast shower simulator for a heterogeneous three-layer calorimeter?

**Senza ID espliciti** (`eval_v3_pair_01_implicit`)

What is the name and generative model family of the fast shower simulator for a heterogeneous three-layer calorimeter?

**Comportamento:** answerable

**Risposta attesa — bozza:** The simulator is CaloGAN, a generative adversarial network (GAN); it is not CaloFlow and this paper does not introduce it as WGAN.

**Criteri:**
- Identify CaloGAN.
- Identify the GAN family; do not substitute WGAN or normalizing flows.

**Evidenze estratte dai testi locali:**

- 1705.02355v2 — Method (sec_002); ruolo: supports_answer

> To alleviate the computational burden of simulating electromagnetic showers, we introduce a method based on Generative Adversarial Networks (GANs) [REF] in order to directly simulate component read-outs in electromagnetic calorimeters.

- 1705.02355v2 — Method (sec_002); ruolo: supports_answer

> This allows the CaloGAN to model the physical sequential dependence among the calorimeter layers.

**Revisione umana:** pending.

## eval_v3_pair_02

**Con fonti esplicite** (`eval_v3_pair_02_explicit`)

Using arXiv:1712.10321v1, in CaloGAN, how is a three-layer calorimeter shower represented as images, and what are the image dimensions of each layer?

**Senza ID espliciti** (`eval_v3_pair_02_implicit`)

In CaloGAN, how is a three-layer calorimeter shower represented as images, and what are the image dimensions of each layer?

**Comportamento:** answerable

**Risposta attesa — bozza:** A shower is represented by three 2D images of per-cell deposited energy, with layer dimensions 3×96, 12×12 and 12×6.

**Criteri:**
- Describe three per-layer images and deposited-energy pixel intensities.
- Give all three dimensions without confusing them with a seven-layer WGAN setup.

**Evidenze estratte dai testi locali:**

- 1712.10321v1 — Dataset (sec_002); ruolo: supports_answer

> The first layer can be represented as a 3 96 image, the middle layer as a 12 12 image, and the last layer as a 12 6 image.

- 1712.10321v1 — Dataset (sec_002); ruolo: supports_answer

> For visualization purposes, a 3-dimensional particle energy signature (Fig. [REF] ) will be displayed in the rest of this paper as a series of three 2D images in - space (Fig. [REF] ), where the pixel intensity represents the sum of the energies of all particles incident to that cell For the purposes of this study, the cell energy is the sum of the energy deposited in the lead and the argon; in practice, only the LAr energy deposits are measured. Dividing out these two components is left for future work (see Sec. [REF] ). .

**Revisione umana:** pending.

## eval_v3_pair_03

**Con fonti esplicite** (`eval_v3_pair_03_explicit`)

Using arXiv:1805.00850v2, which generator architecture is used to translate particle-level hadronic jet images into reconstructed jet images, and how does it connect its encoding and decoding stages?

**Senza ID espliciti** (`eval_v3_pair_03_implicit`)

Which generator architecture is used to translate particle-level hadronic jet images into reconstructed jet images, and how does it connect its encoding and decoding stages?

**Comportamento:** answerable

**Risposta attesa — bozza:** The generator uses a convolutional U-net encoder-decoder with skip connections linking encoding and decoding layers at the same spatial resolution.

**Criteri:**
- Name U-net, not just GAN.
- Explain the encoder-decoder and same-resolution skip connections.

**Evidenze estratte dai testi locali:**

- 1805.00850v2 — Model architecture (sec_004); ruolo: supports_answer

> For the generator we adopt the so called ``U-net'' architecture [REF] , that consists of an encoding section followed by a decoding one, with additional skip connections linking encoding and decoding layers with the same spatial dimension.

**Revisione umana:** pending.

## eval_v3_pair_04

**Con fonti esplicite** (`eval_v3_pair_04_explicit`)

Using arXiv:1807.01954v2, for the WGAN electromagnetic-shower simulator, what speed-up relative to GEANT4 is reported, and how does its evaluation time depend on incident electron energy?

**Senza ID espliciti** (`eval_v3_pair_04_implicit`)

For the WGAN electromagnetic-shower simulator, what speed-up relative to GEANT4 is reported, and how does its evaluation time depend on incident electron energy?

**Comportamento:** answerable

**Risposta attesa — bozza:** The study reports speed-ups up to about three orders of magnitude, depending on hardware; WGAN evaluation time is independent of incident electron energy, unlike GEANT4. This is a setup-specific benchmark, not a universal performance guarantee.

**Criteri:**
- Report the approximate three-order speed-up with hardware/setup qualification.
- Distinguish energy-independent WGAN evaluation from GEANT4 energy scaling.

**Evidenze estratte dai testi locali:**

- 1807.01954v2 — Computational speed-up (sec_016); ruolo: supports_answer

> We observe that fast simulation of electromagnetic showers using this WGAN architecture is up to three orders of magnitude faster than full simulations.

- 1807.01954v2 — Computational speed-up (sec_016); ruolo: supports_answer

> Furthermore, it is remarkable that the time for evaluation of the WGAN is independent of the incident electron energy while it scales significantly in simulations using GEANT4.

**Revisione umana:** pending.

## eval_v3_pair_05

**Con fonti esplicite** (`eval_v3_pair_05_explicit`)

Using arXiv:2106.05285v3, in CaloFlow, what does Flow I generate, what does Flow II model, and why use this two-step design?

**Senza ID espliciti** (`eval_v3_pair_05_implicit`)

In CaloFlow, what does Flow I generate, what does Flow II model, and why use this two-step design?

**Comportamento:** answerable

**Risposta attesa — bozza:** Flow I models layer-wise deposited energies conditioned on incident energy. Flow II models voxel-level shower shapes conditioned on those layer energies and incident energy. The two-step design improves the treatment of energy conservation compared with the tested single-flow alternatives.

**Criteri:**
- Separate layer-energy modeling from voxel/shower-shape modeling.
- State conditioning variables and the energy-conservation motivation.

**Evidenze estratte dai testi locali:**

- 2106.05285v3 — Conclusions (sec_014); ruolo: supports_answer

> To better and more explicitly enforce energy conversation, we invented a 2-step flow, where in the first step we generate the energies deposited in each layer, and in the second step we learn the shower shapes of each layer.

- 2106.05285v3 — Section 4 (sec_004); ruolo: supports_answer

> In this setup, Flow I and Flow II are independent from each other, meaning Flow I can be replaced by an improved version without the need to retrain Flow II, or vice versa.

- 2106.05285v3 — Section 4 (sec_004); ruolo: supports_answer

> Instead, what worked well was a modular, two-step setup in which one NF (which we call ``Flow I'') first learns the distribution of deposited energies conditioned on the input energy, p_1(E_ 0 , E_ 1 , E_ 2 |E_ inc ), Using a normalizing flow for this low-dimensional density estimation problem might be overkill, and simpler alternatives such as KDE [REF] or Mixture Density Networks [REF] might also prove viable. and then another NF (which we call ``Flow II'') learns the shower shapes conditioned on the energies, p_ 2 ( I |E_ 0 , E_ 1 , E_ 2 , E_ inc ).

**Revisione umana:** pending.

## eval_v3_pair_06

**Con fonti esplicite** (`eval_v3_pair_06_explicit`)

Using arXiv:1705.02355v2, arXiv:2106.05285v3, how do CaloGAN and CaloFlow differ in generative modeling approach and access to an explicit likelihood? Support both sides of the comparison.

**Senza ID espliciti** (`eval_v3_pair_06_implicit`)

How do CaloGAN and CaloFlow differ in generative modeling approach and access to an explicit likelihood? Support both sides of the comparison.

**Comportamento:** answerable

**Risposta attesa — bozza:** CaloGAN uses adversarial GAN training and an implicit density without an explicit likelihood. CaloFlow uses normalizing flows with explicit tractable likelihoods and likelihood-based training/model selection. CaloGAN is not itself a normalizing flow.

**Criteri:**
- Attribute GAN/implicit likelihood to CaloGAN.
- Attribute normalizing flows/tractable likelihood to CaloFlow.
- Cite evidence from both studies.

**Evidenze estratte dai testi locali:**

- 1705.02355v2 — Method (sec_002); ruolo: supports_answer

> Though the GAN formulation, by design, does not admit an explicit probability density or explicit likelihood, we gain the ability to sample from the learned generative model in a efficient manner.

- 2106.05285v3 — Conclusions (sec_014); ruolo: supports_answer

> Compared to GANs, they have many advantages, including: explicit, tractable likelihoods; more stable and convergent training and principled model selection (based on the negative log-likelihood).

**Revisione umana:** pending.

## eval_v3_pair_07

**Con fonti esplicite** (`eval_v3_pair_07_explicit`)

Using arXiv:1805.00850v2, arXiv:1807.01954v2, compare the physical objects and detector-level representations in the conditional hadronic-jet image study and the WGAN electromagnetic-shower study. Support both sides of the comparison.

**Senza ID espliciti** (`eval_v3_pair_07_implicit`)

Compare the physical objects and detector-level representations in the conditional hadronic-jet image study and the WGAN electromagnetic-shower study. Support both sides of the comparison.

**Comportamento:** answerable

**Risposta attesa — bozza:** The jet study maps particle-level hadronic-jet images to reconstructed jet images, using CMS samples represented by 32×32 pixels. The WGAN study generates isolated electromagnetic-shower energy depositions in a seven-sensitive-layer sampling calorimeter prototype. Jet images and multilayer electromagnetic showers must not be conflated.

**Criteri:**
- Distinguish hadronic jets from electromagnetic showers.
- Describe particle/reconstructed jet-image mapping versus seven-layer calorimeter showers.
- Cite both studies.

**Evidenze estratte dai testi locali:**

- 1805.00850v2 — Inputs and problem formulation (sec_002); ruolo: supports_answer

> For this study we use simulated samples of hadronic jets produced by the CMS collaboration and published on the CERN open data portal.

- 1805.00850v2 — Inputs and problem formulation (sec_002); ruolo: supports_answer

> We used two sets of hadronic jets: those clustered from the list of stable particles produced by , and those clustered from the list of reconstructed particle candidates.

- 1807.01954v2 — Conclusion (sec_017); ruolo: supports_answer

> In this paper, we presented a method for generating electromagnetic showers using a Wasserstein GAN (WGAN).

- 1807.01954v2 — Conclusion (sec_017); ruolo: supports_answer

> As a concrete example, we followed a prototype setup for a future high-granularity calorimeter with seven sensitive layers featuring about one thousand readout pixels placed in an electron beam.

- 1805.00850v2 — Inputs and problem formulation (sec_002); ruolo: supports_answer

> Jet images are constructed by opening a window of size $ = 0.3 0.3 around the particle-level jet axis. The window is split into 32 32$ identical square pixels and the intensity associated to each pixel is proportional to the total transverse momentum of the jet components contained in it, divided by the transverse momentum of the particle-level jet.

**Revisione umana:** pending.

## eval_v3_pair_08

**Con fonti esplicite** (`eval_v3_pair_08_explicit`)

Using arXiv:1705.02355v2, arXiv:1712.10321v1, how do the initial CaloGAN study and its detailed companion describe the loss enforcing agreement between requested and reconstructed shower energy, and is energy conservation guaranteed exactly? Support both sources.

**Senza ID espliciti** (`eval_v3_pair_08_implicit`)

How do the initial CaloGAN study and its detailed companion describe the loss enforcing agreement between requested and reconstructed shower energy, and is energy conservation guaranteed exactly? Support both sources.

**Comportamento:** answerable

**Risposta attesa — bozza:** Both describe penalizing the absolute difference between requested/nominal and reconstructed energy. The companion specifies a mean absolute error term augmenting the adversarial loss and states that the soft constraint discourages but does not forbid excess deposited energy; it is not an exact hard guarantee.

**Criteri:**
- Explain absolute-energy-deviation penalty in both sources.
- Name mean absolute error in the detailed companion.
- Distinguish soft encouragement from exact conservation; cite both.

**Evidenze estratte dai testi locali:**

- 1705.02355v2 — Method (sec_002); ruolo: supports_answer

> To encourage this to be well modeled, a physics-specific loss component is introduced to penalize absolute deviation between the nominal energy E_0 and the reconstructed energy E .

- 1712.10321v1 — Loss Formulation (sec_005); ruolo: supports_answer

> Note, however, that this formulation discourages, but does not forbid, a deposition of more energy than was requested.

- 1712.10321v1 — Loss Formulation (sec_005); ruolo: supports_answer

> In this work, we augment the classical adversarial loss term L _ adv (Eqn. [REF] ) -- which penalizes the system whenever D fails to classify samples originating from generated or target distributions -- with a mean absolute error term: [EQUATION_002]

**Revisione umana:** pending.

## eval_v3_pair_09

**Con fonti esplicite** (`eval_v3_pair_09_explicit`)

Using arXiv:1705.02355v2, arXiv:1712.10321v1, arXiv:1805.00850v2, arXiv:1807.01954v2, arXiv:2106.05285v3, what exact ATLAS Run 4 production deployment date and detector-wide policy for replacing Geant4 are specified by these fast-simulation studies?

**Senza ID espliciti** (`eval_v3_pair_09_implicit`)

What exact ATLAS Run 4 production deployment date and detector-wide policy for replacing Geant4 are specified by these fast-simulation studies?

**Comportamento:** insufficient_evidence

**Risposta attesa — bozza:** Insufficient evidence: the collection does not establish the requested exact ATLAS Run 4 deployment date or detector-wide replacement policy. Do not turn benchmark results or future-work suggestions into an operational mandate.

**Criteri:**
- Explicitly state that the requested date/policy is not supported.
- Do not invent a date or a collaboration-wide replacement decision.

**Evidenze estratte dai testi locali:**

- 2106.05285v3 — Conclusions (sec_014); ruolo: scope_limitation_not_proof_of_absence

> Further work is needed for realistic setups such as the actual ATLAS/CMS detector or future high-granularity detectors for HL-LHC and ILD, as well as taking into account the incoming angle of the particle and other real-world complications.

**Limite:** Scope excerpts do not prove an exhaustive absence. Human corpus review must confirm the no-answer label.

**Revisione umana:** pending.

## eval_v3_pair_10

**Con fonti esplicite** (`eval_v3_pair_10_explicit`)

Using arXiv:1807.01954v2, arXiv:2106.05285v3, what exact detector-wide production configuration and deployment procedure for replacing Geant4 are prescribed by the WGAN and CaloFlow studies?

**Senza ID espliciti** (`eval_v3_pair_10_implicit`)

What exact detector-wide production configuration and deployment procedure for replacing Geant4 are prescribed by the WGAN and CaloFlow studies?

**Comportamento:** insufficient_evidence

**Risposta attesa — bozza:** Insufficient evidence: the studies describe particular shower-simulation setups and proof-of-concept results, not an exact detector-wide deployment procedure. A seven-layer prototype geometry or a three-layer benchmark is not a detector-wide production prescription.

**Criteri:**
- Abstain from specifying a detector-wide production recipe.
- Distinguish experimental prototype/benchmark geometry from deployment requirements.

**Evidenze estratte dai testi locali:**

- 1807.01954v2 — Conclusion (sec_017); ruolo: scope_limitation_not_proof_of_absence

> While this setup has only been studied in the context of isolated showers, there are ideas for its application to full collision events.

- 2106.05285v3 — Conclusions (sec_014); ruolo: scope_limitation_not_proof_of_absence

> Further work is needed for realistic setups such as the actual ATLAS/CMS detector or future high-granularity detectors for HL-LHC and ILD, as well as taking into account the incoming angle of the particle and other real-world complications.

**Limite:** Scope excerpts do not prove an exhaustive absence. Human corpus review must confirm the no-answer label.

**Revisione umana:** pending.
