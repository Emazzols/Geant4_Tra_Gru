import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

class FirstStepSampler:
    def __init__(self, filename, chunksize=100_000, scalers=None):
        # Lista per accumulare i primi step e energie residue
        first_steps_list = []
        residual_energies = []

        # Lettura a chunk per file grande
        for chunk in pd.read_csv(filename, sep="\t", chunksize=chunksize):
            # Filtro dei primari
            primaries = chunk[chunk["ParentID"] == 0]
            
            # Prendi solo il primo step di ogni evento
            firsts = primaries.groupby("EventNumber").first().reset_index()
            first_steps_list.append(firsts)
            
            # Calcola energia residua per ogni evento
            for event_num, event_data in primaries.groupby("EventNumber"):
                total_edep = event_data["Edep(MeV)"].sum()
                initial_energy = event_data["ParticleEn(MeV)"].iloc[0]
                residual_energy = initial_energy - total_edep
                residual_energies.append(residual_energy)

        # Concatenazione finale dei primi step
        first_steps = pd.concat(first_steps_list, ignore_index=True)

        # Encoding volumi e madri
        le_vol = LabelEncoder()
        le_moth = LabelEncoder()
        first_steps["Volume_enc"] = le_vol.fit_transform(first_steps["VolumeName"])
        first_steps["Mother_enc"] = le_moth.fit_transform(first_steps["MotherVolumeName"])

        # Salva distribuzioni gaussiane per i parametri continui (mu, sigma, sigma_err, min, max)
        self.params = {}
        N = len(first_steps)
        for col in ["Edep(MeV)", "X(mm)", "Y(mm)", "Z(mm)", "Step(mm)"]:
            mu = float(first_steps[col].mean())
            sigma = float(first_steps[col].std())
            sigma_err = sigma / np.sqrt(N) if N > 0 else 0.0
            min_v = float(first_steps[col].min())
            max_v = float(first_steps[col].max())
            self.params[col] = (mu, sigma, sigma_err, min_v, max_v)

        # Memorizza min e max empirici per lo Step (unità fisiche)
        self.step_min = float(first_steps["Step(mm)"].min())
        self.step_max = float(first_steps["Step(mm)"].max())

        # Distribuzioni discrete
        vol_counts = first_steps["Volume_enc"].value_counts(normalize=True)
        moth_counts = first_steps["Mother_enc"].value_counts(normalize=True)

        self.vol_values = vol_counts.index.values
        self.vol_probs = vol_counts.values
        self.moth_values = moth_counts.index.values
        self.moth_probs = moth_counts.values

        # Salva i label encoder
        self.le_vol = le_vol
        self.le_moth = le_moth

        # Distribuzione energia residua (unità fisiche)
        if residual_energies:
            residual_energies = np.array(residual_energies)
            mu = float(residual_energies.mean())
            sigma = float(residual_energies.std())
            sigma_err = float(sigma / np.sqrt(len(residual_energies)))
            min_v = float(residual_energies.min())
            max_v = float(residual_energies.max())
            self.residual_params = (mu, sigma, sigma_err, min_v, max_v)
        else:
            # Fallback
            self.residual_params = (0.015, 0.008, 0.002, 0.0, 0.05)

        # Scaler per normalizzazione
        self.scalers = scalers

        if scalers is not None:
            # Residual energy scalata
            if "ParticleEn(MeV)" in scalers:
                mu, sigma, sigma_err, min_v, max_v = self.residual_params
                mu_s = float(scalers["ParticleEn(MeV)"].transform([[mu]])[0, 0])
                min_s = float(scalers["ParticleEn(MeV)"].transform([[min_v]])[0, 0])
                max_s = float(scalers["ParticleEn(MeV)"].transform([[max_v]])[0, 0])
                self.residual_params_scaled = (mu_s, None, None, min_s, max_s)
            else:
                self.residual_params_scaled = None

            # Step scalato
            if "Step(mm)" in scalers:
                self.step_min_scaled = float(scalers["Step(mm)"].transform([[self.step_min]])[0, 0])
                self.step_max_scaled = float(scalers["Step(mm)"].transform([[self.step_max]])[0, 0])
            else:
                self.step_min_scaled = None
                self.step_max_scaled = None
        else:
            self.residual_params_scaled = None
            self.step_min_scaled = None
            self.step_max_scaled = None

    def _sample_truncated_normal(self, mu, sigma, sigma_err, min_v, max_v, n_sigma=5, max_tries=200):
        """
        Campiona da una normale centrata in mu con sigma_eff limitata,
        troncata a [min_v, max_v].
        """
        if max_v > min_v:
            sigma_cap = (max_v - min_v) / (2 * n_sigma)
            sigma_eff = min(sigma, sigma_cap)
        else:
            sigma_eff = sigma

        val = None
        for _ in range(max_tries):
            cand = np.random.normal(mu, sigma_eff)
            cand += np.random.normal(0.0, sigma_err)
            if min_v <= cand <= max_v:
                val = cand
                break

        if val is None:
            cand = np.random.normal(mu, sigma_eff) + np.random.normal(0.0, sigma_err)
            val = float(np.clip(cand, min_v, max_v))

        return val

    def sample_residual_energy_threshold(self):
        """Campiona una soglia di energia residua nello spazio corretto (scalato se disponibile)."""
        mu, sigma, sigma_err, min_v, max_v = self.residual_params
        val = self._sample_truncated_normal(mu, sigma, sigma_err, min_v, max_v, n_sigma=5)
        if self.scalers is not None and "ParticleEn(MeV)" in self.scalers:
            val = float(self.scalers["ParticleEn(MeV)"].transform([[val]])[0, 0])
        return val

    def sample_first_step(self, device="cpu"):
        first_step = torch.zeros(1, 1, 7, device=device)

        # Parametri continui
        param_names = ["Edep(MeV)", "X(mm)", "Y(mm)", "Z(mm)", "Step(mm)"]
        for i, pname in enumerate(param_names):
            mu, sigma, sigma_err, min_v, max_v = self.params[pname]
            val = self._sample_truncated_normal(mu, sigma, sigma_err, min_v, max_v, n_sigma=5)

            # Protezione specifica per Step
            if pname == "Step(mm)":
                val = float(np.clip(val, self.step_min, self.step_max))

            first_step[0, 0, i] = val

        # Normalizzazione (se scalers presenti)
        if self.scalers is not None:
            for i, col_name in enumerate(["Edep(MeV)", "X(mm)", "Y(mm)", "Z(mm)", "Step(mm)"]):
                if col_name in self.scalers:
                    val_original = first_step[0, 0, i].item()
                    val_normalized = self.scalers[col_name].transform([[val_original]])[0, 0]
                    first_step[0, 0, i] = val_normalized

        # Parametri discreti
        vol_choice = np.random.choice(self.vol_values, p=self.vol_probs)
        moth_choice = np.random.choice(self.moth_values, p=self.moth_probs)
        first_step[0, 0, 5] = vol_choice
        first_step[0, 0, 6] = moth_choice

        return first_step
