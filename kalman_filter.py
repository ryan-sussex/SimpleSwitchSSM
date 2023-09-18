import numpy as np


class KalmanFilter:
    def __init__(
        self,
        n_obs,
        n_hidden,
        transition=None,
        likelihood=None,
        initial_state=None,
        std=0.001,
    ):
        self.n_hidden = n_hidden
        self.n_obs = n_obs

        if initial_state is None:
            self.inital_state = np.zeros(n_hidden)
        else:
            self.inital_state = initial_state

        if transition is None:
            self.transition = np.eye(n_hidden)
        else:
            self.transition = transition

        self.transition_noise = np.eye(n_hidden) * std

        if likelihood is None:
            self.likelihood = np.ones((n_obs, n_hidden))
        else:
            self.likelihood = likelihood

        self.likelihood_noise = np.eye(n_obs) * std

        self.state = self.inital_state

        self.uncertainty = np.eye(n_hidden)

    def sample(self):
        transition_noise = np.random.multivariate_normal(
            np.zeros(self.n_hidden), self.transition_noise
        )
        likelihood_noise = np.random.multivariate_normal(
            np.zeros(self.n_obs),
            self.likelihood_noise,
        )
        self.state = self.transition @ self.state + transition_noise
        obs = self.likelihood @ self.state + likelihood_noise
        print("hidden:", self.state)
        print("obs:", obs)
        return obs

    def forward(self, obs):
        prior = self.transition @ self.state
        prior_uncertainty = (
            self.transition @ self.uncertainty @ self.transition.T
            + self.transition_noise
        )

        residual = obs - self.likelihood @ prior
        residual_uncertainty = (
            self.likelihood @ self.uncertainty @ self.likelihood.T
            + self.likelihood_noise
        )

        kalman_gain = (
            prior_uncertainty
            @ self.likelihood.T
            @ np.linalg.pinv(residual_uncertainty)
        )

        self.state = prior + kalman_gain @ residual

        self.uncertainty_estimate = (
            np.eye(self.n_hidden) - kalman_gain @ self.likelihood
        ) @ prior_uncertainty
        return (self.state, self.uncertainty_estimate)

    def em(self, obs_sequence):
        err = np.zeros((self.n_obs, self.n_hidden))
        state_err = np.zeros((self.n_hidden, self.n_hidden))
        total_cov = np.zeros((self.n_hidden, self.n_hidden))
        prev_state = self.state
        for obs in obs_sequence:
            state, cov = self.forward(obs)
            print("obs", obs)
            print("hidden state", state)
            err += np.expand_dims(obs, 1) @ np.expand_dims(state, 0)
            state_err += np.expand_dims(prev_state, 1) @ np.expand_dims(state, 0)
            total_cov += (
                cov + np.expand_dims(state, 1) @ np.expand_dims(state, 0)
            )
            prev_state = state

        self.likelihood = err @ np.linalg.pinv(total_cov)
        self.transition = state_err @ np.linalg.pinv(total_cov)

    def reset(self):
        self.state = self.inital_state
        self.uncertainty = np.eye(self.n_hidden)


if __name__ == "__main__":
    ref_kf = KalmanFilter(
        n_obs=1,
        n_hidden=1,
        transition=np.array([[1]]),
        likelihood=np.array([[10]]),
        initial_state=np.ones(1),
    )
    kf = KalmanFilter(
        n_obs=1,
        n_hidden=1,
        # transition=np.eye(2),
        # likelihood=np.array([[1, 2], [2, 1]]),
        initial_state=np.ones(1),
    )
    for i in range(1000):
        sample_traj = []
        for i in range(1):
            sample_traj.append(ref_kf.sample())
        print(sample_traj)
        kf.em(sample_traj)
        ref_kf.reset()
        kf.reset()

    print("Parameters")
    print(kf.transition)
    print(kf.likelihood)
