"""
Following https://www.cs.toronto.edu/~hinton/absps/switch.pdf
"""
from typing import Optional, List
import numpy as np
from scipy import special as scisp

from hmm import HMM
from kalman_filter import KalmanFilter


class SwitchedStateSpace:
    def __init__(
            self,
            # n_switches: int,
            # n_hidden: int,
            # n_obs: int,
            state_space_models: Optional[List[KalmanFilter]],
            switching_model: Optional[HMM] = None
        ):
        # self.n_switches = n_switches
        # self.n_obs = n_obs
        # self.n_hidden = n_hidden
        # if state_space_models is not None:
        self.n_switches = len(state_space_models)
        self.n_obs = state_space_models[0].n_hidden
        self.n_hidden = state_space_models[0].n_obs

        # We don't need the observation part of this model
        self.switching_model = HMM(self.n_switches, self.n_switches)
        if switching_model is not None:
            self.switching_model = switching_model

        self.state_space_models = dict()
        for switch in range(self.n_switches):
            self.state_space_models[switch] = state_space_models[switch]

    def sample(self, verbose: bool = True):
        switch = self.switching_model.sample(update_state=True)

        lds_samples = dict()
        for m, lds in self.state_space_models.items():
            lds_samples[m] = lds.sample()

        if verbose:
            print(f"switch state: {switch}")

        return lds_samples[switch]

    def inner_e_steps(self, obs_sequence, switch_probs):
        self.reset()
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
        temp = 1
        print(errors)
        switch_probs = scisp.softmax(temp * errors, axis=1)
        # print(switch_probs)
        obs = np.argmax(switch_probs, axis=1)
        # print(obs)
        # need to introduce SOft evidence
        switch_probs, _, _, = self.switching_model.e_step(obs)
        switch_probs /= switch_probs.sum(axis=1)[:, np.newaxis]
        # print("HERE!")
        # print(switch_probs)
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
        return switch_probs

    def em(self, obs_sequence, iters=10):
        switch_probs = self.e_step(obs_sequence, iters=iters)
        return switch_probs

    def error_per_switch_per_time(self, obs_sequence, ssm_state_post, ssm_cov_post):
        likelihood_matrix = self.get_overall_likelihood_matrices()
        constant_term = np.einsum("to, to -> t", obs_sequence, obs_sequence)
        assert not np.any(constant_term < 0)
        state_term = np.einsum("to, soh, tsh -> ts", obs_sequence, likelihood_matrix, ssm_state_post, )
        cov_term = np.einsum("soh, soj, tshj -> tshj", likelihood_matrix, likelihood_matrix, ssm_cov_post)
        cov_term = np.trace(cov_term, axis1=2, axis2=3)
        # print(cov_term.shape)
        constant_term = constant_term[:, np.newaxis]
        return(- .5 * constant_term + state_term - .5 * cov_term)

    def get_overall_likelihood_matrices(self):
        likelihood = np.zeros((self.n_switches, self.n_hidden, self.n_obs))
        for switch in range(self.n_switches):
            likelihood[switch, :, :] = self.state_space_models[switch]\
                .likelihood
        return likelihood

    def reset(self):
        self.switching_model.reset()
        for switch in range(self.n_switches):
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
    np.set_printoptions(precision=1)

    ref_switch_ssm = SwitchedStateSpace(
        [
            KalmanFilter(n_obs=1, n_hidden=1, transition=np.array([[10]])),
            KalmanFilter(n_obs=1, n_hidden=1, transition=np.array([[0]]))
        ],
        switching_model=HMM(2, 2, transition_prior=np.array([[10, 1],[1, 10]]), emmission_prior=np.array([[99, 1],[1,99]]))
    )

    traj = []
    for i in range(20):
        traj.append(ref_switch_ssm.sample())

    print(np.array(traj))

    posterior_switch = ref_switch_ssm.em(traj, iters=100)
    for i in range(len(posterior_switch)):
        print(np.argmax(posterior_switch[i]))