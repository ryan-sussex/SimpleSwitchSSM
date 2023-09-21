from typing import Optional, List
import numpy as np


class HMM():

    def __init__(
            self, 
            n_hidden: int, 
            n_obs: int,
            transition_prior: Optional[np.array] = None,
            emmission_prior: Optional[np.array] = None
        ) -> None:
        self.n_hidden = n_hidden
        self.n_obs = n_obs
        self.transition_prior = np.ones((n_hidden, n_hidden))
        if transition_prior is not None:
            self.transition_prior = transition_prior
        self.transition_matrix = self.transition_prior / np.sum(self.transition_prior, axis=0)
        self.emission_prior = np.ones((n_obs, n_hidden))
        if emmission_prior is not None:
            self.emission_prior = emmission_prior
        self.emission_matrix = self.emission_prior / self.emission_prior.sum(axis=0)
        self.initial_state = np.ones(n_hidden) / n_hidden
        self.state = self.initial_state

    def sample(self):
        state_probs = self.transition_matrix @ self.state
        state_indx = np.random.choice(self.n_hidden, p=state_probs)
        self.state = np.eye(self.n_hidden)[state_indx, :]
        obs_probs = self.emission_matrix @ self.state
        obs_indx = np.random.choice(self.n_obs, p=obs_probs)
        # obs = np.eye(self.n_obs)[obs_indx, :]
        return obs_indx

    def forward(self, obs_indx):
        # prior = self.transition_matrix @ self.
        likelihood = np.diag(self.emission_matrix[obs_indx, :])
        unnorm_state = likelihood @ self.transition_matrix @ self.state
        self.state = unnorm_state / unnorm_state.sum()
        return self.state

    def em(self, obs_seq: List[np.array]):
        state_dists = []
        empirical_transitions = []
        prev_state = self.initial_state
        for obs in obs_seq:
            state_dist = self.forward(obs)
            state_dists.append(state_dist)
            empirical_transitions.append(
                 np.expand_dims(state_dist, 1) @ 
                 np.expand_dims(prev_state, 0) 
                #  * self.transition_matrix
            )
            prev_state = state_dist
            print("counts", empirical_transitions[-1])

        n_visits = np.sum(state_dists, axis=0)
        n_visits /= n_visits.sum()
        print("n visits", n_visits)
        n_transitions = np.sum(empirical_transitions, axis=0) / n_visits
        print("n transitions", n_transitions)
        self.transition_prior = self.transition_prior + n_transitions
        self.transition_matrix = self.transition_prior / self.transition_prior.sum(axis=0)
        # self.

    def reset(self):
        self.state = self.initial_state



if __name__ == "__main__":

    ref_hmm = HMM(
        n_hidden=2,
        n_obs=2,
        transition_prior=np.array([[10,1], [1,1]]),
        emmission_prior=np.array([[1,0], [0,1]])
    )
    data = []
    for i in range(100):
        data.append(ref_hmm.sample())
    print(data)

    hmm = HMM(
        2, 2,
        emmission_prior=np.array([[1,0], [0,1]])
    )
    # hmm.reset
    print(hmm.state)
    for datum in data:
        hmm.forward(datum)
        print(hmm.state)

    hmm.reset()
    hmm.em(data)
    print(hmm.transition_prior)
    print(hmm.transition_matrix)