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
        self.initial_state = np.eye(n_hidden)[0,:]
        self.state = self.initial_state

    def sample(self, update_state=True):
        state_probs = self.transition_matrix @ self.state
        state_indx = np.random.choice(self.n_hidden, p=state_probs)
        state = np.eye(self.n_hidden)[state_indx, :]
        if update_state:
            self.state = state
        obs_probs = self.emission_matrix @ state
        obs_indx = np.random.choice(self.n_obs, p=obs_probs)
        #
        #  obs = np.eye(self.n_obs)[obs_indx, :]
        return obs_indx

    def forward(self, obs_indx):
        # prior = self.transition_matrix @ self.
        self.state = self._forward(obs_indx, self.state)
        return self.state
    
    def _forward(self, obs_indx, forward_state):
        likelihood = np.diag(self.emission_matrix[obs_indx, :])
        unnorm_state = likelihood @ self.transition_matrix @ forward_state
        state = unnorm_state / unnorm_state.sum()
        return state

    def backward(self, obs_indx, backwards_state):
        likelihood = np.diag(self.emission_matrix[obs_indx, :])
        unnorm_state = backwards_state @ self.transition_matrix @ likelihood 
        state = unnorm_state / unnorm_state.sum()
        return state


    def em(self, obs_seq: List[np.array], lr=.01):
        """
        Online version of baum-welch
        """
        state_dists = []
        empirical_transitions = []
        empirical_emmissions = []
        forwards_state = self.initial_state
        backwards_state = np.ones(self.n_hidden)
        forwards = []
        backwards = []
        # Run forwards - backwards to get posteriors
        for obs in obs_seq:
            forwards_state = self._forward(obs, forwards_state)
            forwards.append(forwards_state)
        forwards = np.array(forwards)

        for obs in reversed(obs_seq):
            backwards_state = self.backward(obs, backwards_state)
            backwards.append(backwards_state)
        backwards = np.array(list(reversed(backwards)))

        state_posteriors = forwards * backwards + 0.001


        # Calculat ML estimates given posteriors
        n_transitions = np.zeros((self.n_hidden, self.n_hidden))
        n_emissions = np.zeros((self.n_obs, self.n_hidden)) 
        for i in range(forwards.shape[0]-1):
            n_transitions += (
                forwards[i][:, np.newaxis] @ backwards[i+1][np.newaxis, :]
                # * self.transition_matrix
            )
            obs_indx = obs_seq[i]
            n_emissions += (
                np.eye(self.n_obs)[obs_indx,:][:, np.newaxis]
                @ state_posteriors[i][np.newaxis, :]
                # / state_posteriors[i]
            )        

        self.transition_prior = self.transition_prior + n_transitions * lr
        self.transition_matrix = self.transition_prior / self.transition_prior.sum(axis=0)
        
        self.emission_prior = self.emission_prior + n_emissions * lr
        self.emission_matrix = self.emission_prior / self.emission_prior.sum(axis=0)
        return


    def em_iterations(self, obs_seq: List[np.array], iters=10):
        for _ in range(iters):
            self.em(obs_seq)
            self.reset()
        
    def reset(self):
        self.state = self.initial_state



if __name__ == "__main__":

    ref_hmm = HMM(
        n_hidden=2,
        n_obs=2,
        transition_prior=np.array([[6,1], [1,7]]),
        emmission_prior=np.array([[99,1], [1,99]])
    )
    data = []
    for i in range(100):
        traj = []
        for i in range(100):
            traj.append(ref_hmm.sample())
            # print(traj)
        data.append(traj)
        ref_hmm.reset()

    test_traj = []
    for i in range(100):
        test_traj.append(ref_hmm.sample())
    


    hmm = HMM(
        2, 2,
        transition_prior=np.array([[1,1], [1,1]]),
        emmission_prior=np.array([[99,1], [1,99]])
    )
    starting_transition = hmm.transition_matrix
    starting_emission = hmm.emission_matrix


    # hmm.reset
    for traj in data:
        # for step in traj:
        hmm.em_iterations(traj, iters=100)
        hmm.reset()

        pred = 0
        error = 0
        for step in test_traj:#
            # print(pred, step)
            error += int(pred != step)
            hmm.forward(step)
            pred = hmm.sample(update_state=False)

        # print("error:", error)
        hmm.reset()

    # print(hmm.transition_prior)
    print("initial transition", starting_transition)
    print("transition probs", hmm.transition_matrix)
    print("gt", ref_hmm.transition_matrix)

    print("\n")
    # print(hmm.emission_prior)
    print("inital emission", starting_emission)
    print("emission probs", hmm.emission_matrix)
    print("gt", ref_hmm.emission_matrix)