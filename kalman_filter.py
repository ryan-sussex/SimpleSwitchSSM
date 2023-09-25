from typing import Optional
import numpy as np


class KalmanFilter:
    """
    See https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.314.2260&rep=rep1&type=pdf
    for ref
    """
    def __init__(
        self,
        n_obs,
        n_hidden,
        transition=None,
        likelihood=None,
        initial_state=None,
        std=1,
    ):
        self.n_hidden = n_hidden
        self.n_obs = n_obs

        if initial_state is None:
            self.inital_state = np.zeros(n_hidden)
        else:
            self.inital_state = initial_state

        if transition is None:
            self.transition = np.ones((n_hidden, n_hidden))
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

    def sample(self, verbose: bool = False):
        transition_noise = np.random.multivariate_normal(
            np.zeros(self.n_hidden), self.transition_noise
        )
        likelihood_noise = np.random.multivariate_normal(
            np.zeros(self.n_obs),
            self.likelihood_noise,
        )
        self.state = self.transition @ self.state + transition_noise
        obs = self.likelihood @ self.state + likelihood_noise
        if verbose:
            print("hidden:", self.state)
            print("obs:", obs)
        return obs

    def _forward(
            self,
            obs: Optional[np.array],
            state: np.array,
            uncertainty
        ):
        prior = self.transition @ state
        prior_uncertainty = (
            self.transition @ uncertainty @ self.transition.T
            + self.transition_noise
        )

        residual = np.zeros(self.n_obs)
        kalman_gain = np.zeros((self.n_hidden, self.n_obs))
        if obs is not None:
            residual = obs - self.likelihood @ prior
            residual_uncertainty = (
                self.likelihood @ uncertainty @ self.likelihood.T
                + self.likelihood_noise
            )

            kalman_gain = (
                prior_uncertainty
                @ self.likelihood.T
                @ np.linalg.pinv(residual_uncertainty)
            )

        state = prior + kalman_gain @ residual

        uncertainty = (
            np.eye(self.n_hidden) - kalman_gain @ self.likelihood
            ) @ prior_uncertainty

        return (state, uncertainty, prior_uncertainty, kalman_gain)

    def backward(
            self,
            state,
            uncertainty,
            prior_uncertainty,
            next_uncertainty,
            next_kalman_gain,
            state_posterior,
            uncertainty_posterior,
            transition_uncertainty,
    ):
        kalman_gain = (
            uncertainty
            @ self.transition.T
            @ np.linalg.pinv(prior_uncertainty)
        )
        state_posterior = (
            state
            + kalman_gain @ (state_posterior - self.transition @ state)
        )
        uncertainty_posterior = (
            uncertainty +
            kalman_gain @ (uncertainty_posterior - prior_uncertainty)
            @ kalman_gain.T
        )
        transition_uncertainty = (
            next_uncertainty @ kalman_gain.T
            + next_kalman_gain
            @ (transition_uncertainty - self.transition @ next_uncertainty)
            @ kalman_gain.T
        )
        return (
            state_posterior,
            uncertainty_posterior,
            transition_uncertainty,
            kalman_gain
        )

    def forward(self, obs: Optional[np.array]):
        self.state, self.uncertainty, _, _ = (
            self._forward(obs, self.state, self.uncertainty)
        )

    def _init_backward(self, final_forward_kalman, uncertainty):
        identity = np.eye(self.n_hidden)
        return (
            (identity - final_forward_kalman @ self.likelihood)
            @ self.transition @ uncertainty
        )

    def e_step(self, obs_sequence):
        filtered_states = []
        filtered_uncertainty = []
        prior_uncertainties = []
        # Forwards (filtering)
        for obs in obs_sequence:
            state, uncertainty, prior_uncertainty, kalman_gain = self._forward(
                obs,
                self.state,
                self.uncertainty
            )
            filtered_states.append(state)
            filtered_uncertainty.append(uncertainty)
            prior_uncertainties.append(prior_uncertainty)
        # Backwards recursions

        state_posterior = filtered_states[-1]
        uncertainty_posterior = filtered_uncertainty[-1]
        state_posteriors = []
        uncertainty_posteriors = []
        transition_posteriors = []

        transition_uncertainty = self._init_backward(
            kalman_gain,
            uncertainty
        )
        for t in reversed(range(len(obs_sequence)-1)):
            (
                state_posterior,
                uncertainty_posterior,
                transition_uncertainty,
                kalman_gain
            ) = self.backward(
                filtered_states[t],
                filtered_uncertainty[t],
                prior_uncertainties[t],
                prior_uncertainties[t+1],
                kalman_gain,
                state_posterior,
                uncertainty_posterior,
                transition_uncertainty
            )
            state_posteriors.append(state_posterior)
            uncertainty_posteriors.append(uncertainty_posterior)
            transition_posteriors.append(transition_uncertainty)

        transition_posteriors = list(reversed(transition_posteriors))
        uncertainty_posteriors = list(reversed(uncertainty_posteriors))
        state_posteriors = list(reversed(state_posteriors))
        return state_posteriors, uncertainty_posteriors, transition_posteriors

    def m_step(self, obs_sequence, state_posteriors, uncertainty_posteriors, transition_posteriors):
        # M step
        state_obs_cov = np.zeros((self.n_obs, self.n_hidden))
        state_cov = np.zeros((self.n_hidden, self.n_hidden))
        transition_cov = np.zeros((self.n_hidden, self.n_hidden))

        for t in range(len(obs_sequence)-1):
            state_obs_cov += obs_sequence[t][:, np.newaxis] @ state_posteriors[t][np.newaxis, :]
            state_cov += (
                state_posteriors[t][:, np.newaxis] @ state_posteriors[t][np.newaxis, :]
                + uncertainty_posteriors[t]
            )
            if t == 0:
                continue
            transition_cov += (
                state_posteriors[t][:, np.newaxis] @ state_posteriors[t-1][np.newaxis, :]
                + transition_posteriors[t]
            )
        # Update parameters
        prev_ll = self.log_likelihood(obs_sequence, state_posteriors)
        state_precision = np.linalg.pinv(state_cov + np.eye(self.n_hidden))
        self.transition = transition_cov @ state_precision
        self.likelihood = state_cov @ state_precision
        new_ll = self.log_likelihood(obs_sequence, state_posteriors)
        return prev_ll, new_ll

    def em(self, obs_sequence):
        (
            state_posteriors,
            uncertainty_posteriors,
            transition_posteriors
        ) = self.e_step(obs_sequence)
        return self.m_step(
            obs_sequence,
            state_posteriors,
            uncertainty_posteriors,
            transition_posteriors
        )

    @staticmethod
    def gaussian_norm(v, mu, cov):
        return (v - mu).T @ cov @ (v - mu)

    def log_likelihood(self, obs, posteriors):
        ll = 0
        for t in range(len(obs) - 2):
            ll += np.linalg.norm(posteriors[t+1] - self.transition @ posteriors[t])
            ll += np.linalg.norm(obs[t] - self.likelihood @ posteriors[t])
        return ll

    def em_iterations(self, obs_sequence, iters, tol = 1e-2):
        for i in range(iters):
            prev_ll, new_ll = self.em(obs_sequence)
            print("iteration: ", i)
            print("previous: ", prev_ll)
            print("new: ", new_ll)
            if abs(new_ll - prev_ll) < tol:
                print("converged")
                break

    def reset(self):
        self.state = self.inital_state
        self.uncertainty = np.eye(self.n_hidden)


if __name__ == "__main__":
    ref_kf = KalmanFilter(
        n_obs=2,
        n_hidden=2,
        transition=np.array([[1, 0], [.1, .9]]),
        likelihood=np.array([[1,0], [0, 1]]),
        initial_state=np.ones(2),
    )
    kf = KalmanFilter(
        n_obs=2,
        n_hidden=2,
        # transition=np.eye(2),
        # likelihood=np.array([[1, 2], [2, 1]]),
        initial_state=np.ones(2),
    )
    initial_transition = kf.transition
    initial_likelihood = kf.likelihood
    for i in range(20):
        sample_traj = []
        for j in range(100):
            sample_traj.append(ref_kf.sample())


        # print(sample_traj)
        kf.em_iterations(sample_traj, iters=10)
        ref_kf.reset()
        kf.reset()

        # if i == 9:
        #     for j in range(5):
        #         print("predictions")
        #         print(kf.forward(None)[0])



    print("Parameters")
    print("initial transition", initial_transition)
    print("transition")
    print(kf.transition)
    print("gt", ref_kf.transition)

    print("initial likelihood", initial_likelihood)
    print("likelihood")
    print(kf.likelihood)
    print("gt", ref_kf.likelihood)
