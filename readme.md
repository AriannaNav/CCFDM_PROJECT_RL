A) Come si runna tutto (train / eval / video / plots)

0) Struttura output (importante)

Con train_ccfdm.py tu generi due alberi paralleli:
	•	Modelli
	•	models/ccfdm/<env_tag>/seed_<seed>/
	•	config.json (hyperparam + env spec usati nel train)
	•	best.pt (checkpoint migliore secondo eval)
	•	best.json (step+score migliore)
	•	last.pt
	•	ckpt_step_000000xxx.pt (ogni --save_every)
	•	Log
	•	logs/ccfdm/<env_tag>/seed_<seed>/
	•	train.jsonl (log training)
	•	eval.jsonl (log valutazioni periodiche)
	•	fig5_eval_curve.png (se fai plots)

<env_tag> è coerente in tutto:
	•	DMC: dmc_<domain>_<task>
	•	MiniGrid: minigrid_<env_id>

1) TRAIN

DMC (esempio)
python main.py train \
  --env dmc --dmc_domain cheetah --dmc_task run \
  --seed 1 --device auto \
  --total_steps 200000 \
  --eval_every 10000 --eval_episodes 10

MiniGrid (esempio)
python main.py train \
  --env minigrid --minigrid_id MiniGrid-Empty-8x8-v0 \
  --seed 1 --device auto \
  --total_steps 200000 

Cosa succede durante il train
	•	Fase “warmup”: fino a --init_random_steps l’azione è random (riempie replay).
	•	Dopo --update_after (default = init_random_steps) e quando replay >= batch_size, parte l’update:
	•	ogni step: update critic (TD) + reward intrinseca
	•	ogni actor_update_freq: update actor+alpha
	•	ogni ccfmd_update_freq: update contrastivo (Eq.8)
	•	ogni critic_target_update_freq: soft update target (Q + encoder)

Checkpoint
	•	best.pt si aggiorna solo quando eval/mean_return migliora.
	•	last.pt sempre aggiornato.

⸻

2) EVAL (carica best.pt e basta) 
python main.py eval --model_dir models/ccfdm/dmc_cheetah_run/seed_1 --episodes 10 --seed 12345

Output:
	•	stampa [EVAL] {mean_return, std_return, ...}
	•	salva eval_result.json dentro model_dir

Nota: il tuo eval attuale è già deterministico, perché usa sempre select_action() (mean action). Il flag --deterministic in eval.py al momento è inutile (vedi sezione “bug/da sistemare”).

⸻
3) VIDEO (rollout e render mp4)
python main.py video \
  --model_dir models/ccfdm/dmc_cheetah_run/seed_1 \
  --ckpt best.pt \
  --out_dir videos \
  --episodes 3 --fps 30 \
  --seed 12345 --deterministic

	•	--deterministic: usa select_action (policy mean)
	•	senza --deterministic: usa sample_action (stocastico)

I file escono tipo:
videos/dmc_cheetah_run_seed12345_ep001.mp4

⸻

4) PLOTS (curva Fig.5 style da eval.jsonl)
python main.py plots --log_dir logs/ccfdm/dmc_cheetah_run/seed_1
crea 
logs/.../fig5_eval_curve.png
B) Come leggere LOG e METRICHE

File di log (JSONL)
	•	train.jsonl: un record per riga, contiene chiavi come train/critic_loss, train/actor_loss, ecc.
	•	eval.jsonl: stesso formato, con eval/mean_return ogni --eval_every.

Ogni riga contiene anche:
	•	step (quando presente)
	•	_time timestamp
	•	_elapsed secondi dal start

Metriche principali

Reward
	•	train/reward_ext_mean: media del reward esterno nel batch RL (quello dell’ambiente)
	•	train/reward_int_mean: media reward intrinseca (curiosity) nel batch
	•	train/reward_total_mean: r_ext + intrinsic_weight * r_int

➡️ Se reward_int_mean domina o “esplode”, spesso l’esplorazione diventa rumorosa e l’extrinsic migliora meno.

SAC
	•	train/critic_loss: MSE TD su Q1+Q2 (più basso non sempre = meglio, ma deve essere stabile)
	•	train/actor_loss: loss policy (alpha*logpi - Q)
	•	train/alpha: temperatura (entropia). Se alpha va troppo su, policy troppo random; troppo giù, policy troppo deterministica presto.

CCFDM
	•	train/contrastive_loss: loss contrastiva Eq.(8) sul “pred next” vs “next target”
	•	Se scende e poi si stabilizza → rappresentazione consistente
	•	Se rimane altissima → o predictor fatica o encoder/aug/temperature non vanno

Episodi
	•	train/episode_return: ritorno totale episodio (solo extrinsic, come lo accumuli nel loop)
	•	train/episode_length: lunghezza episodio

Eval
	•	eval/mean_return, eval/std_return: media e dev std su eval_episodes episodi

⸻

C) Parametri CLI: cosa puoi impostare e che effetto hanno

1) Sistema
	•	--device auto|cpu|mps
	•	mps su Apple Silicon, più veloce ma a volte più “delicato”
	•	--seed N
	•	cambia inizializzazione reti + env train
	•	--deterministic
	•	forza algoritmi deterministici (più riproducibile, spesso più lento)

2) Ambiente
	•	--env dmc|minigrid
	•	DMC: --dmc_domain, --dmc_task, --camera_id
	•	MiniGrid: --minigrid_id
	•	--image_size 84 (crop size usato dal replay)
	•	--frame_stack 3
	•	più stack = più memoria temporale visiva, ma input più grande
	•	--action_repeat 1
	•	aumenta “frame skip” (azioni ripetute). Cambia dinamica, spesso accelera training.
	•	--max_episode_steps
	•	tronca episodi (utile per comparabilità)

3) Training schedule
	•	--total_steps
	•	--init_random_steps
	•	--update_after (default = init_random_steps)
	•	--update_every
	•	quante update gradient per step ambiente (più alto = più compute, più sample-efficiency a volte)
	•	--batch_size
	•	512 è grossa: stabilità ma richiede replay pieno e compute
	•	--replay_size

4) Logging / saving / eval
	•	--log_every stampa e scrive summary
	•	--save_every checkpoint
	•	--eval_every, --eval_episodes = stile paper (10k, 10 ep)

5) Architettura encoder
	•	--hidden_dim
	•	--encoder_feature_dim (z_dim)
	•	--num_layers, --num_filters
	•	più capacità = più compute + rischio overfit, ma spesso meglio su pixel

6) SAC hyperparams
	•	--discount
	•	--critic_tau (EMA per target Q)
	•	--encoder_tau (EMA per target encoder)
	•	--actor_update_freq
	•	--critic_target_update_freq

7) CCFDM / contrastive
	•	--ccfmd_update_freq
	•	quante volte fai Eq.(8). Più alto = più pressione representation/dynamics, ma può “rubare” capacità al RL.
	•	--contrastive_method infonce|triplet|byol
	•	paper: InfoNCE
	•	--temperature (InfoNCE)
	•	più bassa = più “sharp”, può instabilizzare; più alta = più morbida
	•	--normalize / --no_normalize
	•	normalizzare rende similarity più stabile
	•	--triplet_margin (solo triplet)

8) Curiosity / intrinsic
	•	--intrinsic_weight
	•	paper: 0.2. Se troppo alto → agent “esplora per esplorare”
	•	--curiosity_C
	•	scala reward intrinseca (molto impattante)
	•	--curiosity_gamma
	•	decay nel tempo: più alto = intrinsic muore prima, più basso = dura a lungo

⸻
