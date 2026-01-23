import torch
import torch.nn as nn

#ref https://github.com/thanhkaist/CCFDM1/blob/main/encoder.py
# TODO 1) Definisci (opzionale) helper per copiare / tie dei pesi tra moduli
# - serve se vuoi condividere conv tra Actor/Critic oppure inizializzare KE = QE
# - NON usarlo per legare QE e KE durante training EMA

# TODO 2) Crea la classe PixelEncoder
# - __init__(obs_shape, feature_dim, num_layers, num_filters, output_logits=False)
#   * salva obs_shape, feature_dim, num_layers, output_logits
#   * costruisci stack di conv (es. conv1 stride=2, poi conv stride=1)
#   * calcola automaticamente flatten_dim con un dummy forward (niente OUT_DIM hard-coded)
#   * definisci fc (flatten_dim -> feature_dim) + LayerNorm(feature_dim)
#
# - forward_conv(obs)
#   * input: obs float o uint8 (B,C,H,W)
#   * normalizza in [0,1] se arriva uint8 o comunque /255.0
#   * passa nelle conv con ReLU
#   * flatten -> (B, flatten_dim)
#
# - forward(obs, detach=False)
#   * h = forward_conv(obs)
#   * se detach: h = h.detach()
#   * h = fc(h) -> ln(h)
#   * se output_logits: return ln_out
#     altrimenti: return torch.tanh(ln_out)
#
# - copy_conv_weights_from(source)
#   * copia SOLO i pesi delle conv (utile per weight tying actor/critic)
#
# TODO 3) (Opzionale) Crea la classe IdentityEncoder
# - per observation già vettoriali: forward ritorna obs (eventuale cast/reshape)
# - feature_dim = obs_dim

# TODO 4) (Opzionale) Factory make_encoder
# - mappa stringa -> classe encoder (es. {"pixel": PixelEncoder, "identity": IdentityEncoder})
# - ritorna l’istanza corretta in base a encoder_type

# TODO 5) (Nel tuo progetto CCFDM) definisci come userai QE e KE
# - in ccfdm_agent.py istanzi: qe = PixelEncoder(...), ke = PixelEncoder(...)
# - inizializza ke con i pesi di qe (copia) una volta
# - aggiorna ke SOLO via EMA (non tramite optimizer)