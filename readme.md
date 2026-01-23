


## 📁 Struttura del progetto (flat)

Il progetto utilizza una **struttura flat** (tutti file Python allo stesso livello)  
per evitare problemi di import e mantenere la massima semplicità operativa.
CCFDM_PROJECT_RL/
│
├── train_ccfdm.py      # entry point del training
├── eval.py             # valutazione e metriche
│
├── ccfdm_agent.py      # algoritmo CCFDM (CUORE DEL PAPER)
├── sac.py              # Soft Actor-Critic puro (baseline)
│
├── encoder.py          # Query / Key Encoder
├── models.py           # Actor, Critic, Action Embedding, FDM
├── losses.py           # InfoNCE + Curiosity (intrinsic reward)
│
├── data.py             # replay buffer + data augmentation
│
├── make_env.py         # factory ambienti
├── dmc.py              # wrapper DeepMind Control Suite
├── minigrid.py         # wrapper MiniGrid
│
├── utils.py            # seed, device, helper comuni
├── logger.py           # logging centralizzato
│
└── readme.md

---

## 🧠 Algoritmo CCFDM

### `ccfdm_agent.py` — **CUORE DEL PAPER**

Questo file implementa **l’intero algoritmo CCFDM**  
così come descritto nel paper (**Algorithm 1**).

### Responsabilità
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
- q = QE(ô_t)
- k = KE(ô_t)
- k⁺ = KE(ô_{t+1})

4. **Predizione dinamica**
- q' = FDM(q, AE(a_t))

5. **Loss contrastiva (InfoNCE, Eq. 8)**
- positiva: (q', k⁺)
- negative: altri sample nel batch

6. **Intrinsic reward (Eq. 9)**
- errore del Forward Dynamics Model
- normalizzazione task-agnostic
- decay temporale

7. **Reward finale**
8. **Update**
- encoder
- action embedding
- FDM
- SAC (actor + critic)

9. **Update EMA**
---

## 🤖 Reinforcement Learning

### `sac.py`

Implementazione **pura** di Soft Actor-Critic.
-ACTOR 
-CRITIC

**Caratteristiche**
- 2 Q-networks
- update actor
- update temperatura α
- soft update dei target network

⚠️ **Non conosce nulla di CCFDM, curiosità o contrastive learning**  
Serve come:
- baseline
- componente riutilizzata da `ccfdm_agent.py`

---

## 🧩 Modelli

### `encoder.py`
- CNN per osservazioni RGB
- produce embedding latente `z`
- supporto `detach`
- Query Encoder (QE)
- Key Encoder (KE) aggiornato solo via EMA

---

### `models.py`
Contiene:
- Action Embedding (AE)
- Forward Dynamics Model (FDM)

Il FDM:
- input: `[z_t, e(a_t)]`
- output: `ẑ_{t+1}`
- supervisione **implicita** tramite loss contrastiva

---


## 📉 Loss e Curiosità

### `losses.py`

Contiene:
- **InfoNCE** (Eq. 8)
- **Curiosity Module** (Eq. 9)

Responsabilità:
- costruzione logits contrastivi
- similarità (dot / bilinear)
- cross-entropy
- errore FDM
- normalizzazione e decay temporale

❌ Non conosce SAC né il replay buffer.

---

## 🛠 Utility

---

### `logger.py`
- logging centralizzato
- scalari
- supporto TensorBoard (opzionale)

---

## 🚀 Training ed Evaluation

### `train_ccfdm.py`
Entry point principale.

Responsabilità:
- parsing argomenti
- setup device e seed
- creazione env
- creazione agent
- training loop
- evaluation periodica
- salvataggio modelli

⚠️ **Nessuna logica algoritmica qui**

---

### `eval.py`
Valutazione separata:
- return vs environment steps
- sample efficiency (100k / 500k)
- state-space coverage
- policy stability

---

## 🔧 Training parametrico e sperimentazione

Il progetto è progettato per consentire **esperimenti controllati** senza riscrivere codice.

È possibile:
- cambiare ambiente (GridWorld → MiniGrid → DMC)
- cambiare loss contrastiva
- disattivare curiosità o FDM
- confrontare:
- SAC only
- CURL only
- CCFDM completo

Il file `ccfdm_agent.py` rimane **immutato**:  
le variazioni avvengono **per composizione**, non per riscrittura.

---

## 📊 Metriche di analisi

- **Sample Efficiency**
- return vs steps
- score @ 100k / 500k
- **State-Space Coverage**
- dispersione embedding QE
- celle visitate (GridWorld)
- entropy embedding (DMC)
- **Policy Stability**
- varianza dei ritorni
- entropy della policy
- oscillazioni di α

---

## ✅ Obiettivo finale

Una **riproduzione fedele del paper CCFDM**, con:
- codice leggibile
- struttura coerente
- generalizzazione cross-task
- base solida per ricerca e tesi

## Cosa Installare
- pip install gymnasium minigrid
- pip install dm_control
- pip install pillow
- pip install opencv-python