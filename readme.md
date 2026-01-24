
Curiosity Contrastive Forward Dynamics Model (CCFDM)

This repository contains an implementation of Curiosity Contrastive Forward Dynamics Model (CCFDM) built on top of Soft Actor-Critic (SAC), following:

Nguyen et al., “Sample-efficient Reinforcement Learning Representation Learning with Curiosity Contrastive Forward Dynamics Model”, 2021

The framework combines:
	•	pixel-based representation learning,
	•	contrastive learning with a momentum encoder,
	•	a forward dynamics model in latent space,
	•	intrinsic motivation based on prediction error,
	•	off-policy RL (SAC).

⸻

1. Features
	•	End-to-end training from pixels
	•	Contrastive representation learning (InfoNCE)
	•	Forward dynamics model for temporal consistency
	•	Curiosity-driven intrinsic reward
	•	Compatible with DeepMind Control Suite and MiniGrid
	•	Deterministic evaluation and video rendering

 2. Project Structure
 
├── main.py              # training entry point
├── train_ccfdm.py       # legacy training script
├── eval.py              # evaluation from saved checkpoint
├── plots.py             # plot learning curves
├── video.py             # render rollout video
├── ccfdm_agent.py       # SAC + CCFDM agent
├── ccfdm_modules.py     # FDM, action embedding, contrastive module
├── encoder.py           # pixel encoder
├── sac.py               # SAC implementation
├── data.py              # replay buffer
├── dmc.py               # DeepMind Control wrapper
├── minigrid_env.py      # MiniGrid wrapper
├── make_env.py          # environment factory
├── losses.py            # contrastive and curiosity losses
├── utils.py             # utilities (seed, soft update, etc.)
└── logger.py            # logging utilities


Training corto (20k step)

Serve solo a verificare che tutto funzioni.
python train_ccfdm.py \
  --env dmc \
  --dmc_domain walker \
  --dmc_task walk \
  --seed 1 \
  --device mps \
  --total_steps 20000 \
  --init_random_steps 2000 \
  --update_after 1000 \
  --eval_every 5000 \
  --eval_episodes 5

📁 Output atteso:
models/ccfdm/dmc_walker_walk/seed_1/
  ├── last.pt
  ├── best.pt
logs/ccfdm/dmc_walker_walk/seed_1/
Se non vedi errori e vengono salvati i file → sei a posto.


3️⃣ Training “vero” (paper-like)
python train_ccfdm.py \
  --env dmc \
  --dmc_domain walker \
  --dmc_task walk \
  --seed 1 \
  --device mps \
  --total_steps 500000 \
  --batch_size 512 \
  --eval_every 10000 \
  --eval_episodes 10 \
  --save_every 10000


💡 Altri task DMC validi:
	•	cartpole swingup
	•	finger spin
	•	cheetah run
	•	ball_in_cup catch
	•	reacher easy

4️⃣ Evaluation (policy deterministica)

Metodo diretto
python eval.py \
  --model_dir models/ccfdm/dmc_walker_walk/seed_1 \
  --episodes 10 \
  --device mps

Output:
	•	mean return
	•	std return

5️⃣ Rendering / Video
python video.py \
  --model_dir models/ccfdm/dmc_walker_walk/seed_1 \
  --out_dir videos \
  --episodes 3 \
  --fps 30 \
  --device mps

📁 Output:
videos/
  └── dmc_walker_walk_seed1_ep0.mp4

Se .mp4 non viene scritto:

pip install imageio-ffmpeg

6️⃣ Tutto da main.py (come volevi tu)

✔️ Eval + Render insieme
python main.py --eval --render \
  --model_dir models/ccfdm/dmc_walker_walk/seed_1 \
  --device mps

✔️ Modalità subcommand (più pulita)
python main.py run --do_eval --do_video \
  --model_dir models/ccfdm/dmc_walker_walk/seed_1 \
  --device mps


⸻

7️⃣ Plot delle curve (stile Fig.5)
python plots.py \
  --log_dir logs/ccfdm/dmc_walker_walk/seed_1

Output:
logs/.../fig5_eval_curve.png

8️⃣ Workflow consigliato (ordine giusto)
	1.	✅ Training corto (20k) → verifica che tutto gira
	2.	🚀 Training lungo (500k)
	3.	📊 Eval (eval.py)
	4.	🎥 Video (video.py)
	5.	📈 Plot (plots.py)

9️⃣ Note importanti (da ricercatrice a ricercatrice)
	•	L’intrinsic reward è attiva solo in training
	•	Eval e video usano policy deterministica
	•	Data augmentation non è più un no-op su DMC 84×84
	•	CCFDM è paper-faithful (Eq.8 + Eq.9, decay singolo)