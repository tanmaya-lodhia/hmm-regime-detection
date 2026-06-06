"""
model.py — Fit a Gaussian HMM to SPX return features and decode hidden states.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


N_STATES = 3
RANDOM_SEED = 42


class RegimeHMM:
    """
    Wraps GaussianHMM to detect bull / bear / high-vol regimes.

    Design decisions:
      - GaussianHMM: each hidden state emits observations drawn from a
        multivariate Gaussian.  Since our features (log_return, rolling_vol)
        are roughly bell-shaped within a regime, this is a natural fit.
      - covariance_type="full": each state gets its own full covariance matrix
        so the model captures correlations between return and vol within a state.
        "diag" would be faster but assumes features are independent — wrong here
        because in a bear regime, returns AND vol move together.
      - n_iter=200: EM (Expectation-Maximization) iterates until convergence.
        200 is more than enough for 3 states; default 10 often under-converges.
      - StandardScaler: the two features live on very different scales
        (return ~0.001, vol ~0.01).  Scaling to zero-mean unit-variance stops
        the EM from being dominated by whichever feature has larger magnitude.
    """

    def __init__(self, n_states: int = N_STATES, random_seed: int = RANDOM_SEED):
        self.n_states = n_states
        self.scaler = StandardScaler()
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=random_seed,
            verbose=False,
        )
        self.state_map: dict[int, str] = {}  # raw state index -> label

    def fit(self, X: np.ndarray) -> "RegimeHMM":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._label_states(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw integer state sequence."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_named(self, X: np.ndarray) -> list[str]:
        """Return human-readable regime labels."""
        raw = self.predict(X)
        return [self.state_map[s] for s in raw]

    # ------------------------------------------------------------------
    # State labelling
    # ------------------------------------------------------------------

    def _label_states(self, X: np.ndarray):
        """
        After fitting, HMM states are just integers 0/1/2 — the model has no
        idea which one is 'bull'.  We label them post-hoc by looking at the
        mean features of each state.

        Labelling rule (applied in order):
          1. State with highest mean log_return  -> 'bull'
          2. State with lowest mean log_return   -> 'bear'
          3. Remaining state                     -> 'high_vol'

        This is simple and interpretable.  An alternative is to sort by mean
        vol — both work; return-based sorting is more intuitive for a trader.
        """
        raw_states = self.predict(X)
        state_means = {}
        for s in range(self.n_states):
            mask = raw_states == s
            state_means[s] = X[mask, 0].mean()  # mean log_return

        sorted_states = sorted(state_means, key=state_means.get, reverse=True)
        self.state_map = {
            sorted_states[0]: "bull",
            sorted_states[1]: "high_vol",
            sorted_states[2]: "bear",
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self, X: np.ndarray) -> pd.DataFrame:
        """Per-state statistics: mean return, mean vol, frequency."""
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

    def transition_matrix(self) -> pd.DataFrame:
        """Return the learned transition matrix with labelled rows/cols."""
        labels = [self.state_map[i] for i in range(self.n_states)]
        return pd.DataFrame(
            self.model.transmat_,
            index=labels,
            columns=labels,
        ).round(4)


if __name__ == "__main__":
    from data import fetch_spx, build_features

    raw = fetch_spx()
    X, df = build_features(raw)

    hmm = RegimeHMM().fit(X)

    print("\n=== State Summary ===")
    print(hmm.summary(X))

    print("\n=== Transition Matrix ===")
    print(hmm.transition_matrix())
