import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


N_STATES = 3
RANDOM_SEED = 42


class RegimeHMM:

    def __init__(self, n_states=N_STATES, random_seed=RANDOM_SEED):
        self.n_states = n_states
        self.scaler = StandardScaler()
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=random_seed,
            verbose=False,
        )
        self.state_map = {}

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._label_states(X)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_named(self, X):
        return [self.state_map[s] for s in self.predict(X)]

    def _label_states(self, X):
        # label states by mean return: highest = bull, lowest = bear
        raw = self.predict(X)
        means = {s: X[raw == s, 0].mean() for s in range(self.n_states)}
        ranked = sorted(means, key=means.get, reverse=True)
        self.state_map = {
            ranked[0]: "bull",
            ranked[1]: "high_vol",
            ranked[2]: "bear",
        }

    def summary(self, X):
        raw = self.predict(X)
        rows = []
        for s in range(self.n_states):
            mask = raw == s
            rows.append({
                "state":       self.state_map[s],
                "count":       int(mask.sum()),
                "pct":         f"{mask.mean()*100:.1f}%",
                "mean_return": f"{X[mask, 0].mean()*100:.4f}%",
                "mean_vol":    f"{X[mask, 1].mean()*100:.4f}%",
            })
        return pd.DataFrame(rows).set_index("state")

    def transition_matrix(self):
        labels = [self.state_map[i] for i in range(self.n_states)]
        return pd.DataFrame(self.model.transmat_, index=labels, columns=labels).round(4)


if __name__ == "__main__":
    from data import fetch_spx, build_features

    raw = fetch_spx()
    X, df = build_features(raw)
    hmm = RegimeHMM().fit(X)

    print(hmm.summary(X))
    print(hmm.transition_matrix())
