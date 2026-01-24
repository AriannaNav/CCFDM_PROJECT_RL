
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


⸻

3. Training

DeepMind Control Suite

python main.py \
  --env dmc \
  --dmc_domain walker \
  --dmc_task walk \
  --seed 1 \
  --total_steps 500000 \
  --eval_every 10000 \
  --eval_episodes 10 \
  --batch_size 128

Other supported tasks:
	•	finger spin
	•	cartpole swingup
	•	cheetah run
	•	ball_in_cup catch
	•	reacher easy

MiniGrid

python main.py \
  --env minigrid \
  --minigrid_id MiniGrid-Empty-8x8-v0 \
  --seed 1 \
  --total_steps 200000


⸻

4. Evaluation

To evaluate a trained agent:

python eval.py \
  --model_path models/ccfdm/<env>/<seed>/best.pt

5. Plotting Learning Curves

To reproduce evaluation curves (Fig. 5–style):
python plots.py \
  --logdir logs/ccfdm \
  --env walker_walk

The script aggregates multiple seeds and plots evaluation return vs environment steps.

6. Video Rendering

To render a rollout from a trained agent:
python video.py \
  --model_path models/ccfdm/<env>/<seed>/best.pt \
  --output video.mp4
  The video is generated using the deterministic policy (no exploration noise).

8. Reproducibility
	•	All experiments are seed-controlled
	•	Best and last checkpoints are saved automatically
	•	Configuration is stored alongside each run


## Cosa Installare
- pip install gymnasium minigrid
- pip install dm_control
- pip install pillow
- pip install opencv-python