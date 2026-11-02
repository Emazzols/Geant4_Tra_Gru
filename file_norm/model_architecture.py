"""
Costruiamo il modello

- modello ibrido GRU (Gated Recurrent Unit) + Transformer con chunking adattivo
- il chunking serve per dividere le sequenze e passare alla rete solo un certo numero di step per volta. Sostanzialmente è simile al concetto del minibatch, ma mentre il minibatch suddivide gli eventi, il chunk suddivide gli step dell'evento
- La GRU è una rete ricorrente che usa due porte — update e reset — per decidere quanto mantenere o dimenticare dello stato precedente, aggiornando così lo stato nascosto in modo efficiente e prevenendo il problema del vanishing gradient su sequenze lunghe.
- Il Transformer usa meccanismi di self-attention per pesare dinamicamente l'importanza di ogni elemento della sequenza rispetto agli altri, permettendo di catturare dipendenze a lungo raggio in parallelo, senza ricorrere a passaggi sequenziali come nelle RNN.
- C'è anche la parte di generazione della sequenza, che usa il modello addestrato per generare nuove sequenze step by step, con vincoli fisici e campionamento stocastico per le variabili categoriche.
- E' necessario che la parte di generazione sia integrata nel modello per garantire che lo stato nascosto della GRU venga mantenuto correttamente tra i vari step generati, e che quindi non si abbia ripetizione di pattern o perdita di coerenza nella sequenza generata.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from first_step_distributions import FirstStepSampler


class GRUTransformerGenerator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_volume_classes,
        n_mother_volume_classes,
        energy_dim=1,
        gru_layers=1,
        nhead=4,
        max_chunk_len=100,
        num_transformer_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_chunk_len = max_chunk_len  # Chunk size fisso a 100 step

        # Proiezione iniziale
        self.input_proj = nn.Linear(input_dim + energy_dim, hidden_dim)

        # GRU con stato persistente
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
        )

        # Transformer Encoder (senza positional encoding esplicita)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)

        # Output heads
        # n_cont = 5: Edep, X, Y, Z, Step
        # ora prediciamo MU e LOGVAR per ciascuna variabile continua => 2 * n_cont
        self.n_cont = 5
        self.output_cont = nn.Linear(hidden_dim, self.n_cont * 2)  # mu + logvar
        self.output_vol = nn.Linear(hidden_dim, n_volume_classes)
        self.output_mother = nn.Linear(hidden_dim, n_mother_volume_classes)

        # Nota: nel file originale avevi la possibilità di istanziare il sampler qui
        # self.first_step_sampler = FirstStepSampler("file.txt")
        # di solito viene assegnato esternamente o fornito con scalers nel notebook.

    def forward(self, input_seq, energy, mask=None, hidden_state=None):
        """Processa la sequenza a chunk di max_chunk_len step"""
        batch_size, seq_len, _ = input_seq.shape

        # Prepara l'energia (se è 1-D -> espandi a [...,1])
        if energy.dim() == 1:
            energy = energy.unsqueeze(-1)
        energy_expanded = energy.unsqueeze(1)  # Per espansione nei chunk

        all_outputs = []
        for start_idx in range(0, seq_len, self.max_chunk_len):
            end_idx = min(start_idx + self.max_chunk_len, seq_len)

            # Estrai chunk corrente
            chunk = input_seq[:, start_idx:end_idx, :]
            chunk_mask = mask[:, start_idx:end_idx] if mask is not None else None

            chunk_output, hidden_state = self._process_chunk(
                chunk=chunk,
                energy=energy_expanded.expand(-1, chunk.size(1), -1),
                mask=chunk_mask,
                hidden_state=hidden_state,
            )
            all_outputs.append(chunk_output)

        # Ricostruisci l'output completo
        return torch.cat(all_outputs, dim=1), hidden_state

    def _process_chunk(self, chunk, energy, mask, hidden_state=None):
        """Processa un singolo chunk"""
        # Concatena input + energia
        x = torch.cat([chunk, energy], dim=-1)
        x = self.input_proj(x)

        # GRU con stato nascosto (automaticamente gestito tra i chunk)
        gru_out, hidden_state = self.gru(x, hidden_state)

        # Transformer: PyTorch si aspetta src_key_padding_mask=True per PAD (da ignorare)
        transformer_mask = ~mask if mask is not None else None
        trans_out = self.transformer(gru_out, src_key_padding_mask=transformer_mask)

        # Output heads
        # out_cont: [B, S, 2*n_cont] -> mu + logvar
        out_cont = self.output_cont(trans_out)
        out_vol = self.output_vol(trans_out)
        out_mother = self.output_mother(trans_out)

        # concateniamo nello stesso ordine che usavi prima:
        # [mu, logvar, vol_logits, mother_logits]
        return torch.cat([out_cont, out_vol, out_mother], dim=-1), hidden_state

    # -----------------------------------------------------------
    '''
    Generatrice step-by-step (usata dal notebook di generazione)
    - energy: valore (fisico o normalizzato a seconda del tuo flusso). Qui manteniamo la logica di conversione
      e la soglia come nel tuo originale (vedi commenti più sotto).
    - max_len: lunghezza massima
    - temperature: per il sampling categoriale (0 -> deterministico)
    - noise_scale: fattore moltiplicativo applicato alla sigma predetta (se 0 => uso mu deterministico)
    - seed: seed opzionale per riproducibilità
    - force_energy_conservation: applica la logica della soglia campionata e forza l'ultimo step a consumare il residuo
    '''
    def generate_sequence(
        self, energy, max_len=100, temperature=1.0, noise_scale=0.02, seed=None, force_energy_conservation=True
    ):
        device = next(self.parameters()).device
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # --------------------------------------------------------------
        # 1) Energia iniziale: può essere fornita in unità fisiche (ParticleEn MeV)
        #    o già in spazio normalizzato secondo come la passi dal notebook.
        #    Qui seguo la tua logica originale: se esiste first_step_sampler.scalers
        #    provo a convertire in spazio della rete (Edep-space) per inizializzare residual_energy.
        # --------------------------------------------------------------
        energy_phys = float(energy) if not torch.is_tensor(energy) else float(energy.item())

        # Tensore energia per la rete (forma compatibile con il resto del codice)
        energy_tensor = torch.tensor([energy], device=device, dtype=torch.float32)

        # ------------------------------
        # 2) Residual energy nello spazio di Edep (coerente con la soglia)
        # ------------------------------
        if (
            hasattr(self, "first_step_sampler")
            and getattr(self.first_step_sampler, "scalers", None) is not None
            and "Edep(MeV)" in self.first_step_sampler.scalers
        ):
            # assumo energy è in ParticleEn (fisico); trasformo nello spazio Edep normalizzato
            try:
                residual_energy = float(
                    self.first_step_sampler.scalers["Edep(MeV)"].transform([[energy_phys]])[0, 0]
                )
            except Exception:
                residual_energy = float(energy_phys)
        else:
            residual_energy = float(energy_phys)

        # ------------------------------
        # 3) Energy threshold: converti in Edep-space prima dell'uso
        # ------------------------------
        if force_energy_conservation and hasattr(self, "first_step_sampler"):
            thr_sample = self.first_step_sampler.sample_residual_energy_threshold()
            # thr_sample potrebbe essere in ParticleEn; cerchiamo di riportarlo in Edep-space
            if (
                getattr(self.first_step_sampler, "scalers", None) is not None
                and "ParticleEn(MeV)" in self.first_step_sampler.scalers
            ):
                try:
                    thr_phys = self.first_step_sampler.scalers["ParticleEn(MeV)"].inverse_transform(
                        [[thr_sample]]
                    )[0, 0]
                except Exception:
                    thr_phys = float(thr_sample)
                if "Edep(MeV)" in self.first_step_sampler.scalers:
                    try:
                        energy_threshold = float(
                            self.first_step_sampler.scalers["Edep(MeV)"].transform([[thr_phys]])[0, 0]
                        )
                    except Exception:
                        energy_threshold = float(thr_phys)
                else:
                    energy_threshold = thr_phys
            else:
                energy_threshold = float(thr_sample)
        else:
            # fallback (unità coerente con Edep-space)
            energy_threshold = 0.01

        # ------------------------------
        # 4) Primo step iniziale (campionato dal sampler)
        # ------------------------------
        if not hasattr(self, "first_step_sampler"):
            raise RuntimeError("first_step_sampler non trovato: assegnalo (FirstStepSampler) prima di generare.")
        current_step = self._initialize_first_step(energy_tensor, device)  # [1,1,7] nel formato atteso

        generated_steps = []

        # ------------------------------
        # 5) Loop autoregressivo
        # ------------------------------
        hidden_state = None
        for t in range(max_len):
            with torch.no_grad():
                next_step, hidden_state = self._generate_next_step(
                    current_step,
                    energy_tensor,
                    hidden_state,
                    temperature,
                    residual_energy,
                    noise_scale,
                )

                # estrai energia predetta (prima colonna dei continui)
                # next_step shape: [1,1,7] -> next_step[0,0,0]
                current_edep = float(next_step[0, 0, 0].item())

                # Usa la soglia (tutte le grandezze sono ora nello stesso spazio Edep)
                if force_energy_conservation and (residual_energy - current_edep) < energy_threshold:
                    # forza ultimo step a consumare il residuo
                    next_step[0, 0, 0] = torch.tensor(
                        residual_energy, device=next_step.device, dtype=next_step.dtype
                    )
                    current_edep = residual_energy
                    generated_steps.append(next_step.squeeze(0).squeeze(0))
                    break

                generated_steps.append(next_step.squeeze(0).squeeze(0))
                residual_energy -= current_edep

                if residual_energy <= 0 or t >= max_len - 1:
                    break

                current_step = next_step

        return torch.stack(generated_steps) if generated_steps else current_step.squeeze(0)

    # -----------------------------------------------------------
    '''
    Inizializza il primo step della sequenza
    - Energia depositata con il primo step
    - posizione del primo deposito
    - lunghezza del primo step
    - volumi iniziali
    - tutti i valori sono campionati gaussianamente dai dati reali (c'è un altro file che se ne occupa)
    - aggiunge un piccolo rumore gaussiano basato sull'errore statistico
    '''
    def _initialize_first_step(self, energy, device):
        # la FirstStepSampler fornisce un tensore [1,1,7] già normalizzato (se scalers sono presenti)
        return self.first_step_sampler.sample_first_step(device)

    # -----------------------------------------------------------
    '''
    Genera lo step successivo della sequenza e aggiorna la memoria a lungo termine
    - aggiunge rumore secondo il logvar predetto (heteroscedastic)
    - applica le costrizioni fisiche
    - il campionamento categorico serve ad aumentare il peso dei volumi più improbabili (aumenta la casualità rendendo il modello meno conservativo)
    '''
    def _generate_next_step(self, current_step, energy, hidden_state, temperature, residual_energy=None, noise_scale=0.02):
        # Assicurati che energy sia almeno 1-D
        if energy.dim() == 0:
            energy = energy.unsqueeze(0)
        if energy.dim() == 1:
            energy = energy.unsqueeze(-1)

        # current_step: [batch=1, seq_len=1, features=7] (continui + vol_idx + mother_idx)
        chunk_output, new_hidden = self._process_chunk(
            chunk=current_step,
            energy=energy.unsqueeze(1).expand(-1, current_step.size(1), -1),
            mask=torch.ones(current_step.size(0), current_step.size(1), dtype=torch.bool, device=current_step.device),
            hidden_state=hidden_state,
        )

        # chunk_output: [B, seq_len, 2*n_cont + n_vol + n_moth]
        # Estraiamo la parte continua (mu + logvar)
        cont_out_pair = chunk_output[:, :, : (self.n_cont * 2)]  # [B, seq, 2*n_cont]
        vol_logits = chunk_output[:, :, (self.n_cont * 2) : (self.n_cont * 2 + self.output_vol.out_features)]
        mother_logits = chunk_output[:, :, (self.n_cont * 2 + self.output_vol.out_features) :]

        # split mean / logvar
        mu, logvar = torch.chunk(cont_out_pair, 2, dim=-1)  # entrambi [B, seq, n_cont]

        # stabilità numerica per logvar
        logvar = torch.clamp(logvar, min=-12.0, max=12.0)
        sigma = torch.exp(0.5 * logvar)  # sigma in scala della rete

        # campionamento gaussiano: noise_scale scala la sigma predetta.
        if noise_scale is None:
            noise_scale = 1.0
        if noise_scale > 0.0:
            eps = torch.randn_like(sigma)
            cont_sample = mu + (sigma * noise_scale) * eps
        else:
            # comportamento deterministico (use mu)
            cont_sample = mu

        # Applica vincoli fisici
        cont_sample = self._apply_physics_constraints(cont_sample, current_step, residual_energy)

        # Campionamento categorico robusto (gestisce temperature == 0)
        vol_class_idx = self._sample_categorical(vol_logits, temperature)  # [B, seq, 1] int64
        mother_class_idx = self._sample_categorical(mother_logits, temperature)  # [B, seq, 1]

        # Convertiamo gli indici in float per mantenerli nello stesso formato del first_step
        vol_idx_float = vol_class_idx.to(dtype=cont_sample.dtype)
        moth_idx_float = mother_class_idx.to(dtype=cont_sample.dtype)

        # Costruiamo next_step: continui (n_cont) + vol_idx (1) + mother_idx (1)
        # cont_sample ha shape [B, seq, n_cont]; prendere l'ultimo frame (qui seq==1)
        next_step = torch.cat([cont_sample, vol_idx_float, moth_idx_float], dim=-1)  # [B, seq, n_cont+1+1]

        return next_step, new_hidden

    # -----------------------------------------------------------
    '''
    In base al valore di temperatura scelto, dà più peso a volumi meno probabili ma fisicamente possibili
    - temperature <= 0 -> comportamento deterministico (argmax)
    '''
    def _sample_categorical(self, logits, temperature):
        """
        logits: [B, seq, n_classes]
        return: indices [B, seq, 1] (dtype long)
        """
        # sicurezza: logits potrebbe avere forma [1,1,n]
        if temperature is None:
            temperature = 1.0

        # se temperature <= 0: deterministico
        if float(temperature) <= 0.0:
            idx = torch.argmax(logits, dim=-1, keepdim=True)  # [B, seq, 1]
            return idx.long()

        # altrimenti campioniamo dalla categorical
        # evitiamo divisione per esatti zero con una small epsilon
        temp = float(temperature)
        scaled = logits / temp
        probs = F.softmax(scaled, dim=-1)
        # torch.multinomial richiede shape [N, n_classes], quindi flatten batch*seq
        B, S, C = probs.shape
        probs_flat = probs.view(-1, C)
        # aggiungiamo una piccola epsilon per stabilità
        probs_flat = probs_flat.clamp(min=1e-12)
        idx_flat = torch.multinomial(probs_flat, 1)  # [B*S, 1]
        idx = idx_flat.view(B, S, 1)
        return idx.long()

    # -----------------------------------------------------------
    '''
    Applica alcuni vincoli fisici
    - l'energia depositata deve essere minore o uguale a quella residua
    - la lunghezza degli step deve essere compresa tra il minimo e il massimo degli step reali nel file di dati
    '''
    def _apply_physics_constraints(self, features, prev_step, residual_energy=None):
        """
        features: [B, seq, n_cont] (in scala della rete)
        prev_step: tensore con lo stesso formato dei step utilizzati in input (usato per contesti)
        residual_energy: valore scalato coerente con lo spazio Edep della rete (float) oppure None
        """
        features = features.clone()

        # --- Edep: 1° elemento in features[...,0]
        features[..., 0] = torch.clamp(features[..., 0], min=0.0)

        # Se ho una residual_energy (float), imposto un upper bound
        if residual_energy is not None:
            try:
                max_edep = float(residual_energy)
            except Exception:
                max_edep = None

            if max_edep is not None:
                features[..., 0] = torch.clamp(features[..., 0], max=max_edep)

        # --- Step: 5° elemento in features[...,4]
        if hasattr(self, "first_step_sampler") and hasattr(self.first_step_sampler, "step_min"):
            min_step = float(self.first_step_sampler.step_min)
            max_step = float(self.first_step_sampler.step_max)

            if min_step == max_step:
                features[..., 4] = torch.clamp(features[..., 4], min=min_step, max=max_step)
            else:
                features[..., 4] = torch.clamp(features[..., 4], min=min_step, max=max_step)
        else:
            features[..., 4] = torch.clamp(features[..., 4], min=1e-9)

        return features

    # -----------------------------------------------------------
    '''
    E' la funzione generatrice che viene richiamata nel notebook 
    '''
    def generate_multiple_particles(self, energy, n_particles=5, max_len=100, temperature=1.0, noise_scale=0.02):
        particles = []
        for i in range(n_particles):
            seed = torch.randint(0, 100000, (1,)).item() + i * 1000
            particle = self.generate_sequence(
                energy, max_len=max_len, temperature=temperature, noise_scale=noise_scale, seed=seed, force_energy_conservation=True
            )
            particles.append(particle)
        return particles
