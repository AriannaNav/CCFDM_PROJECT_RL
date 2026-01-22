🔹 agents/

agents/sac.py

Responsabilità
	•	Implementazione pura di Soft Actor-Critic
	•	NON contiene CCFDM
	•	Serve come:
	•	baseline
	•	modulo riutilizzato da ccfdm_agent.py

Contenuto
	•	update critic (2 Q-networks)
	•	update actor
	•	update temperatura α
	•	soft update target networks
	•	logging di:
	•	actor loss
	•	critic loss
	•	entropy
	•	α

⚠️ Non deve sapere nulla di contrastive learning o curiosità

⸻

agents/ccfdm_agent.py

CUORE DEL PAPER

Questo file orchestra l’intero algoritmo CCFDM (Algorithm 1 del paper).

Responsabilità
	•	sampling dal replay buffer
	•	data augmentation
	•	encoding:
	•	Query Encoder (QE)
	•	Key Encoder (KE, EMA)
	•	Forward Dynamics Model (FDM)
	•	loss contrastiva (InfoNCE)
	•	calcolo intrinsic reward (Eq. 9)
	•	combinazione reward estrinseco + intrinseco
	•	chiamata a SAC update

Pipeline da implementare (step-by-step)
	1.	Sample batch B = (o_t, a_t, o_{t+1}, r_t)
	2.	Applica augmentation → B̂
	3.	Calcola:
	•	q = QE(ô_t)
	•	k = KE(ô_t)
	•	k⁺ = KE(ô_{t+1})
	4.	Predizione dinamica:
	•	q' = FDM(q, AE(a_t))
	5.	Loss contrastiva InfoNCE:
	•	positiva: (q', k⁺)
	•	negative: batch
	6.	Intrinsic reward:
	•	errore FDM
	•	normalizzazione
	•	decay temporale
	7.	Reward finale:
    r_total = r_ext + C * exp(-γt) * r_int
    8.	Update:
	•	encoder
	•	action embedding
	•	FDM
	•	SAC (actor, critic)
	9.	Update EMA:
	•	KE ← τ·QE + (1−τ)·KE
    🔹 data/

data/replay_buffer.py

Responsabilità
	•	replay buffer unico per tutti gli env
	•	supporto:
	•	immagini
	•	azioni continue
	•	rewards
	•	obs_next
	•	supporto a batch per contrastive learning
	•	anchor
	•	positive
	•	negative (implicitamente batch)

⚠️ NON inserire logica di training qui.

data/augmentations.py

Responsabilità
	•	data augmentation per immagini:
	•	random crop
	•	shift
	•	color jitter (opzionale)
	•	deve essere usata solo per contrastive learning, non per env.step

🔹 envs/

envs/make_env.py

Factory centrale degli ambienti

Qui si decide:
	•	quale env usare (dmc, minigrid, gridworld)
	•	wrapper comuni
	•	output standardizzato

Output obbligatorio per TUTTI gli env
	•	obs: uint8 [C, 84, 84]
	•	action: float32 (anche se discreto internamente)
	•	reward: float
	•	done

Questo è ciò che permette un’unica codebase.

envs/dmc.py

Wrapper per:
	•	DeepMind Control Suite
	•	pixel observations
	•	continuous actions

envs/minigrid.py

Wrapper per:
	•	MiniGrid
	•	mapping azioni discrete → continue
	•	rendering RGB
	•	frame stacking

🔹 losses/

losses/contrastive.py

Loss InfoNCE (Eq. 8 del paper)

Responsabilità:
	•	costruzione logits
	•	similarità (dot o bilinear)
	•	cross-entropy

Questo file non conosce SAC, env, reward.

Estendibile:
	•	puoi aggiungere altre loss (BYOL, SupCon, ecc.)

⸻

losses/intrinsic.py

Curiosity Module (Eq. 9)

Responsabilità:
	•	calcolo errore FDM
	•	normalizzazione (task-agnostic)
	•	decay temporale
	•	clipping

NON deve accedere al replay buffer.

⸻

🔹 models/

models/encoder.py

Query Encoder / Key Encoder
	•	CNN per immagini
	•	output embedding z
	•	supporto detach
	•	KE viene aggiornato via EMA (non gradienti)

⸻

models/action_embed.py

Action Embedding (AE)
	•	MLP
	•	a_t → e(a_t)
	•	concat con z_t

⸻

models/fdm.py

Forward Dynamics Model (FDM)
	•	input: [z_t, e(a_t)]
	•	output: ẑ_{t+1}
	•	loss: implicita via contrastive objective

⸻

models/actor.py, models/critic.py

Architettura SAC standard.

⸻

🔹 scripts/

scripts/train_ccfdm.py

Entry point principale

Responsabilità:
	•	parsing config
	•	creazione env
	•	creazione agent
	•	training loop
	•	evaluation periodica
	•	logging
	•	salvataggio modelli

⚠️ Qui NON va logica algoritmica, solo orchestrazione.

⸻

scripts/eval.py

Valutazione separata:
	•	return vs step
	•	sample efficiency (100k / 500k)
	•	state-space coverage (embedding-based)
	•	policy stability (varianza ritorni)

⸻

🔹 utils/

utils/ema.py

Aggiornamento:
θ_k ← τ θ_q + (1 − τ) θ_k

utils/logger.py
	•	TensorBoard
	•	CSV / JSON
	•	logging centralizzato

⸻

utils/seed.py

Riproducibilità:
	•	torch
	•	numpy
	•	env

⸻

📊 Metriche da implementare

Sample Efficiency
	•	return vs environment steps
	•	score @ 100k, 500k

State-Space Coverage
	•	embedding QE
	•	clustering o dispersione
	•	GridWorld: celle visitate
	•	DMC: embedding entropy

Policy Stability
	•	varianza ritorni
	•	entropy policy
	•	oscillazioni α

⸻

🔬 Estensioni previste
	•	nuove contrastive loss (losses/)
	•	nuovi env (envs/)
	•	ablation:
	•	no FDM
	•	no intrinsic reward
	•	CURL only
	•	SAC only

⸻

✅ Obiettivo finale

Una riproduzione fedele del paper, con:
	•	struttura chiara
	•	estendibilità
	•	confronti scientifici
	•	generalizzazione cross-task
