import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np


class FirstStepSampler:
    def __init__(self, filename, chunksize=100_000):
        # Leggo i dati a chunk
        reader = pd.read_csv(filename, sep="\t", chunksize=chunksize)

        first_steps_list = []
        residual_energies = []

        for chunk in reader:
            # seleziono solo le tracce primarie
            primaries = chunk[chunk["ParentID"] == 0]

            # primo step di ogni evento
            first_steps = primaries.groupby("EventNumber").first()
            first_steps_list.append(first_steps)

            # calcolo energie residue per ogni evento
            total_edep = primaries.groupby("EventNumber")["Edep(MeV)"].sum()
            initial_energy = primaries.groupby("EventNumber")["ParticleEn(MeV)"].first()
            residual_energy = initial_energy - total_edep
            residual_energies.extend(residual_energy.values)

        # concateno tutti i primi step
        first_steps = pd.concat(first_steps_list)

        # codifico i volumi
        self.le_vol = LabelEncoder()
        self.le_moth = LabelEncoder()
        first_steps["Volume_enc"] = self.le_vol.fit_transform(first_steps["VolumeName"])
        first_steps["Mother_enc"] = self.le_moth.fit_transform(first_steps["MotherVolumeName"])

        # Parametri continui: mantengo anche i vettori empirici
        self.params = {}
        self.empirical_values = {}   # mappa col -> np.array dei valori osservati
        N = len(first_steps)

        # Se il dataset è molto grande, posso prendere un sottocampione per risparmiare memoria
        MAX_SAMPLES_PER_PARAM = 200_000

        for col in ["Edep(MeV)", "X(mm)", "Y(mm)", "Z(mm)", "Step(mm)"]:
            vals = first_steps[col].values
            if len(vals) > MAX_SAMPLES_PER_PARAM:
                # sotto-campiono senza rimpiazzo per mantenere la frequenza relativa
                idx = np.random.choice(len(vals), size=MAX_SAMPLES_PER_PARAM, replace=False)
                vals_sampled = vals[idx]
            else:
                vals_sampled = vals.copy()

            mu = float(vals_sampled.mean())
            sigma = float(vals_sampled.std()) if len(vals_sampled) > 1 else 0.0
            sigma_err = sigma / np.sqrt(N) if N > 0 else 0.0
            min_v = float(vals_sampled.min())
            max_v = float(vals_sampled.max())

            self.params[col] = (mu, sigma, sigma_err, min_v, max_v)
            self.empirical_values[col] = vals_sampled  # salvo i valori empirici

        # min/max dello step
        self.step_min = first_steps["Step(mm)"].min()
        self.step_max = first_steps["Step(mm)"].max()

        # distribuzioni empiriche dei volumi
        vol_counts = first_steps["Volume_enc"].value_counts(normalize=True)
        self.vol_values = vol_counts.index.values
        self.vol_probs = vol_counts.values

        moth_counts = first_steps["Mother_enc"].value_counts(normalize=True)
        self.moth_values = moth_counts.index.values
        self.moth_probs = moth_counts.values

        # energia residua: salvo anche i parametri
        if residual_energies:
            residuals = np.array(residual_energies)
            mu = residuals.mean()
            sigma = residuals.std()
            sigma_err = sigma / np.sqrt(len(residuals))
            min_v = residuals.min()
            max_v = residuals.max()
            self.residual_params = (mu, sigma, sigma_err, min_v, max_v)
            self.residual_values = residuals
        else:
            # fallback
            self.residual_params = (0.015, 0.008, 0.002, 0.0, 0.05)
            self.residual_values = np.array([0.015])

    def sample_residual_energy_threshold(self):
        # campionamento empirico dell'energia residua
        return float(np.random.choice(self.residual_values))

    def sample_first_step(self, device="cpu"):
        first_step = torch.zeros(1, 1, 7, device=device)

        # Parametri continui: campionamento empirico con frequenze reali
        param_names = ["Edep(MeV)", "X(mm)", "Y(mm)", "Z(mm)", "Step(mm)"]
        for i, pname in enumerate(param_names):
            vals_emp = self.empirical_values[pname]
            val = float(np.random.choice(vals_emp))

            # Protezione per Step
            if pname == "Step(mm)":
                val = float(np.clip(val, self.step_min, self.step_max))

            first_step[0, 0, i] = val

        # Parametri discreti: già basati sulla distribuzione empirica
        vol_choice = np.random.choice(self.vol_values, p=self.vol_probs)
        moth_choice = np.random.choice(self.moth_values, p=self.moth_probs)
        first_step[0, 0, 5] = vol_choice
        first_step[0, 0, 6] = moth_choice

        return first_step