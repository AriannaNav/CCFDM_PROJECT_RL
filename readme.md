# CCFDM – Curiosity Contrastive Forward Dynamics Model (SAC)

Riproduzione **fedele, modulare e sperimentale** del paper  
**“Curiosity Contrastive Forward Dynamics Model (CCFDM)”**,  
implementata in **PyTorch**, utilizzando **Soft Actor-Critic (SAC)** come algoritmo RL di base  
(**NON PPO**).

L’obiettivo del progetto è:
- riprodurre **esattamente l’algoritmo del paper**
- mantenere una **struttura pulita e modulare**
- supportare **ambienti diversi** (GridWorld → MiniGrid → DMC)
- permettere **ablation study, confronti e generalizzazione**

---
---

## 🔹 agents/

### agents/sac.py

**Responsabilità**
- Implementazione **pura** di Soft Actor-Critic
- **NON** contiene CCFDM
- Serve come:
  - baseline sperimentale
  - modulo riutilizzato da `ccfdm_agent.py`

**Contenuto**
- update critic (2 Q-networks)
- update actor
- update temperatura α
- soft update dei target networks
- logging di:
  - actor loss
  - critic loss
  - entropy
  - α

⚠️ **Non deve sapere nulla di contrastive learning o curiosità**

---

### agents/ccfdm_agent.py

## 🧠 CUORE DEL PAPER

Questo file orchestra **l’intero algoritmo CCFDM**  
(**Algorithm 1 del paper**).

**Responsabilità**
- sampling dal replay buffer
- data augmentation
- encoding:
  - Query Encoder (QE)
  - Key Encoder (KE, aggiornato via EMA)
- Forward Dynamics Model (FDM)
- loss contrastiva (InfoNCE)
- calcolo intrinsic reward (Eq. 9)
- combinazione reward estrinseco + intrinseco
- chiamata agli update SAC

---

### 🔁 Pipeline algoritmica (step-by-step)

1. **Sample batch**
2. **Data augmentation**
3. **Encoding**
- q = QE(ô_t)
- k = KE(ô_t)
- k⁺ = KE(ô_{t+1})

4. **Predizione dinamica**
- q' = FDM(q, AE(a_t))

5. **Loss contrastiva (InfoNCE)**
- positiva: (q', k⁺)
- negative: altri sample nel batch

6. **Intrinsic reward**
- errore FDM
- normalizzazione
- decay temporale

7. **Reward finale**
8. **Update**
- encoder
- action embedding
- FDM
- SAC (actor, critic)

9. **Update EMA**
---

## 🔹 data/

### data/replay_buffer.py

**Responsabilità**
- replay buffer **unico per tutti gli env**
- supporto a:
- immagini
- azioni continue
- reward
- obs_next
- supporto batch per contrastive learning:
- anchor
- positive
- negative (implicitamente il batch)

⚠️ **NON inserire logica di training qui**

---

### data/augmentations.py

**Responsabilità**
- data augmentation per immagini:
- random crop
- shift
- color jitter (opzionale)
- deve essere usata **solo per contrastive learning**
- **NON** va usata per `env.step`

---

## 🔹 envs/

### envs/make_env.py

**Factory centrale degli ambienti**

Qui si decide:
- quale env usare (dmc, minigrid, gridworld)
- wrapper comuni
- output standardizzato

**Output obbligatorio per TUTTI gli env**
- obs: uint8 `[C, 84, 84]`
- action: float32 (anche se discreto internamente)
- reward: float
- done: bool

👉 Questo è ciò che permette **un’unica codebase**.

---

### envs/dmc.py

Wrapper per:
- DeepMind Control Suite
- osservazioni pixel
- azioni continue

---

### envs/minigrid.py

Wrapper per:
- MiniGrid
- mapping azioni discrete → continue
- rendering RGB
- frame stacking

---

## 🔹 losses/

### losses/contrastive.py

**Loss InfoNCE (Eq. 8 del paper)**

**Responsabilità**
- costruzione logits
- similarità (dot product o bilinear)
- cross-entropy

❌ Questo file **non conosce** SAC, env o reward.

**Estendibile**
- BYOL
- SupCon
- altre loss contrastive

---

### losses/intrinsic.py

**Curiosity Module (Eq. 9 del paper)**

**Responsabilità**
- calcolo errore FDM
- normalizzazione (task-agnostic)
- decay temporale
- clipping

❌ NON deve accedere al replay buffer.

---

## 🔹 models/

### models/encoder.py

Query Encoder / Key Encoder
- CNN per immagini
- output embedding `z`
- supporto `detach`
- KE aggiornato **solo via EMA** (no gradienti)

---

### models/action_embed.py

Action Embedding (AE)
- MLP
- a_t → e(a_t)
- concatenazione con z_t

---

### models/fdm.py

Forward Dynamics Model (FDM)
- input: `[z_t, e(a_t)]`
- output: `ẑ_{t+1}`
- loss **implicita** tramite contrastive objective

---

### models/actor.py  
### models/critic.py

Architettura **SAC standard**.

---

## 🔹 scripts/

### scripts/train_ccfdm.py

**Entry point principale**

**Responsabilità**
- parsing config
- creazione env
- creazione agent
- training loop
- evaluation periodica
- logging
- salvataggio modelli

⚠️ **QUI NON VA LOGICA ALGORITMICA**

---

### scripts/eval.py

Valutazione separata:
- return vs step
- sample efficiency (100k / 500k)
- state-space coverage
- policy stability

---

## 🔹 utils/

### utils/ema.py

Aggiornamento EMA:
---

### utils/logger.py
- TensorBoard
- CSV / JSON
- logging centralizzato

---

### utils/seed.py
- riproducibilità
- torch
- numpy
- env

---

## 📊 Metriche da implementare

### Sample Efficiency
- return vs environment steps
- score @ 100k, 500k

### State-Space Coverage
- embedding QE
- clustering / dispersione
- GridWorld: celle visitate
- DMC: entropy dell’embedding

### Policy Stability
- varianza ritorni
- entropy policy
- oscillazioni di α

---

## 🔬 Estensioni previste

- nuove contrastive loss (`losses/`)
- nuovi env (`envs/`)
- ablation study:
  - no FDM
  - no intrinsic reward
  - CURL only
  - SAC only

---
## 🔧 Training parametrico e modularità sperimentale

L’intero progetto è progettato per supportare **training parametrico e sperimentazione controllata**, senza modificare la struttura dell’algoritmo CCFDM.

Tutti gli elementi **variabili** del training sono isolati e configurabili:
- **ambienti** (GridWorld, MiniGrid, DMC) tramite `envs/make_env.py`
- **loss contrastive** tramite moduli intercambiabili in `losses/`
- **curiosity / intrinsic reward** tramite `losses/intrinsic.py`
- **iperparametri di training** (τ, γ, C, batch size, encoder dim, ecc.) tramite config o argomenti di script

Questo permette di:
- confrontare **diverse contrastive loss** (InfoNCE, BYOL-style, SupCon) **a parità di agente**
- testare la **generalizzazione cross-task** usando la stessa architettura
- passare da ambienti semplici (GridWorld) a complessi (DMC) **senza cambiare encoder o agente**
- eseguire **ablation study** disattivando singoli moduli (FDM, intrinsic reward, EMA, ecc.)

Il file `agents/ccfdm_agent.py` rimane **immutato**:  
ogni variazione sperimentale avviene **per composizione**, non per riscrittura del codice.

Questo approccio garantisce:
- riproducibilità
- confronti equi
- estensibilità del framework
- aderenza rigorosa al paper originale

## 🔧 Come realizzare training parametrico e confronti sperimentali

La modularità del progetto non è solo concettuale, ma **realizzata a livello di codice** tramite una separazione netta tra:
- **orchestrazione** (script)
- **algoritmo** (agent)
- **componenti sostituibili** (env, loss, modelli)

### Ambienti
Tutti gli ambienti sono creati tramite una **factory unica** (`envs/make_env.py`).
Ogni nuovo ambiente (GridWorld, MiniGrid, DMC, ecc.) deve:
- restituire osservazioni RGB normalizzate (`uint8 [C, 84, 84]`)
- esporre azioni come `float32`, anche se discrete internamente
- rispettare la stessa interfaccia `reset / step`

In questo modo **l’agent e l’encoder non cambiano mai**, indipendentemente dal task.

### Loss contrastive
Le loss contrastive sono isolate nella cartella `losses/`.
Ogni loss è implementata come modulo indipendente (es. `InfoNCE`, `BYOL-style`, `SupCon`)
e può essere selezionata:
- tramite flag/config
- oppure istanziata dinamicamente in `ccfdm_agent.py`

Il training loop resta invariato: cambia solo **la funzione di loss**, non la pipeline.

### Curiosity e intrinsic reward
Il calcolo dell’intrinsic reward è separato in `losses/intrinsic.py`.
Questo permette di:
- modificare normalizzazione, decay o clipping
- disattivare completamente la curiosità
- confrontare CCFDM con baseline (SAC / CURL-only)

senza introdurre dipendenze con il replay buffer o l’ambiente.

### Training loop
Gli script in `scripts/` (es. `train_ccfdm.py`) contengono **solo**:
- parsing dei parametri
- creazione di env, agent e logger
- gestione del loop di training ed evaluation

Tutta la logica algoritmica rimane confinata in `agents/ccfdm_agent.py`.

Questo design consente di effettuare:
- ablation study
- confronti tra loss
- generalizzazione cross-environment
- scaling progressivo della difficoltà del task

**senza riscrivere codice e senza introdurre bias strutturali.**

## ✅ Obiettivo finale

Una **riproduzione fedele del paper**, con:
- struttura chiara
- separazione netta dei ruoli
- generalizzazione cross-task
- confronti sperimentali solidi
  
