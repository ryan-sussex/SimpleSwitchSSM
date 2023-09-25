"""
Following https://www.cs.toronto.edu/~hinton/absps/switch.pdf
"""
import numpy as np

from hmm import HMM
from kalman_filter import KalmanFilter


class SwitchedStateSpace:
    def __init__(self, n_switches: int, n_hidden: int, n_obs: int):
        self.n_switches = n_switches
        self.n_obs = n_obs
        self.n_hidden = n_hidden

        # We don't need the observation part of this model
        self.switching_model = HMM(self.n_switches, self.n_switches)
        self.state_space_models = dict()
        for switch in range(n_switches):
            self.state_space_models[switch] = KalmanFilter(n_obs, n_hidden)

    def sample(self, verbose: bool = True):
        switch = self.switching_model.sample(update_state=True)

        lds_samples = dict()
        for m, lds in self.state_space_models.items():
            lds_samples[m] = lds.sample()

        if verbose:
            print(f"switch state: {switch}")

        return lds_samples[switch]

    def inner_e_steps(self, obs_sequence, switch_probs):
        weighted_data = np.einsum("to, ts -> tso", obs_sequence, switch_probs)
        ssm_posteriors = np.zeros(
            (len(obs_sequence), self.n_switches, self.n_hidden)
        )
        ssm_covariance_posteriors = np.zeros(
            (len(obs_sequence), self.n_switches, self.n_hidden, self.n_hidden)
        )
        for switch in range(self.n_switches):
            (
                state_posteriors,
                uncertainty_posteriors,
                transition_posteriors
            ) = self.state_space_models[switch].e_step(weighted_data[:, switch, :])
            ssm_posteriors[:-1, switch, :] = np.array(
                state_posteriors
            )
            ssm_covariance_posteriors[:-1, switch, :, :] = np.array(
                uncertainty_posteriors
            )
        # get switch probs
        errors = self.error_per_switch_per_time(obs_sequence, ssm_posteriors, ssm_covariance_posteriors)
        switch_probs = errors / errors.sum(axis=1)[:, np.newaxis]
        return switch_probs

    def e_step(self, obs_sequence, iters: int):
        # E Step
        # Need to seperate e and m steps in individual models
        # Compute prediction errors for each chain
        obs_sequence = np.array(obs_sequence)  # t x n_obs
        init_switch_probs = np.ones((len(obs_sequence), self.n_switches))
        init_switch_probs /= init_switch_probs.sum(axis=1)[:, np.newaxis]

        switch_probs = init_switch_probs
        for i in range(iters):
            switch_probs = self.inner_e_steps(obs_sequence, switch_probs)
            print(switch_probs)

    def em(self, obs_sequence):
        self.e_step(obs_sequence, iters=3)

    def error_per_switch_per_time(self, obs_sequence, ssm_state_post, ssm_cov_post):
        likelihood_matrix = self.get_overall_likelihood_matrices()
        print(ssm_state_post.shape)
        print(likelihood_matrix.shape)
        constant_term = np.einsum("to, to -> t", obs_sequence, obs_sequence)
        state_term = np.einsum("to, soh, tsh -> ts", obs_sequence, likelihood_matrix, ssm_state_post, )
        print(ssm_cov_post.shape)
        cov_term = np.einsum("soh, soh, tshj -> ts", likelihood_matrix, likelihood_matrix, ssm_cov_post)
        # cov_term = np.trace(axis1=1, axis2=2)
        constant_term = constant_term[:, np.newaxis]
        return constant_term + state_term + cov_term

    def get_overall_likelihood_matrices(self):
        likelihood = np.zeros((self.n_switches, self.n_hidden, self.n_obs))
        for switch in range(self.n_switches):
            likelihood[switch, :, :] = self.state_space_models[switch]\
                .likelihood
        return likelihood

    def reset(self):
        self.switching_model.reset()
        for switch in range(self, self.n_switches):
            self.state_space_models[switch].reset()

        # for obs in obs_sequence:
        #     prediction_error = np.array(self.n_switches)
        #     for m, lds in self.state_space_models.items():
        #         prediction_error[m] = lds.error()
        #     errors.append(prediction_error)
        #     # Compute hmm posteriors (given prediction errors)
        #     self.switching_model.em(errors)
        #     # Run kalman smoother weighted by hidden states
        #     for m, lds in self.state_space_models.items():
        #         data = obs * self.switching_model.posterior[m]
        #         lds.em(data)

        # M step


if __name__ == "__main__":
    ref_switch_ssm = SwitchedStateSpace(n_switches=2, n_hidden=1, n_obs=1)

    traj = []
    for i in range(3):
        traj.append(ref_switch_ssm.sample())

    print(traj)

    ref_switch_ssm.em(traj)
