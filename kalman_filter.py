import numpy as np



class KalmanFilter():

    def __init__(
            self, 
            n_obs,
            n_hidden,
            transition = None,
            likelihood = None,
            initial_state = None
        ):
        self.n_hidden = n_hidden
        self.n_obs = n_obs

        if initial_state is None:
            self.inital_state = np.zeros(
                n_hidden
            )
        else:
            self.inital_state = initial_state

        if transition is None:
            self.transition = np.zeros(
                (n_hidden, n_hidden)
            )
        else:
            self.transition = transition


        self.transition_noise = np.eye(n_hidden) 

        if likelihood is None:
            self.likelihood = np.ones(
                (n_obs, n_hidden)
            )
        else:
            self.likelihood = likelihood

        self.likelihood_noise = np.eye(n_obs)

        self.state = self.inital_state

        self.uncertainty = np.eye(n_hidden)


    def sample(self):
        transition_noise = np.random.multivariate_normal(
            np.zeros(self.n_hidden),
            self.transition_noise
        )
        likelihood_noise = np.random.multivariate_normal(
            np.zeros(self.n_obs),
            self.likelihood_noise,
        )
        self.state = self.transition @ self.state + transition_noise
        obs = self.likelihood @ self.state + likelihood_noise
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

        kalman_gain = prior_uncertainty @ self.likelihood.T @ np.linalg.pinv(residual_uncertainty)

        self.state = prior + kalman_gain @ residual
        
        
        self.uncertainty_estimate = (
            (np.eye(self.n_hidden) - kalman_gain @ self.likelihood) 
            @ prior_uncertainty 
        )
        return (self.state, self.uncertainty_estimate)
    
    def em(self, obs_sequence):
        err = np.zeros((self.n_obs, self.n_hidden))
        total_cov = np.zeros((self.n_hidden, self.n_hidden))
        for obs in obs_sequence:
            state, cov = self.forward(obs)
            err += np.expand_dims(obs, 1) @ np.expand_dims(state, 0) 
            total_cov += cov
        
        self.likelihood = err @ np.linalg.pinv(total_cov)

    
    def reset(self):
        self.state = self.s_zero
        self.uncertainty = np.eye(self.n_hidden)



if __name__ == "__main__":
    

    ref_kf = KalmanFilter(
        n_obs=2, 
        n_hidden=2,
        transition=np.eye(2),
        likelihood=np.array(
            [
                [1, 2],
                [1, 1]
            ]
        ),
        initial_state=np.ones(2)
    )

    sample_traj = [] 
    for i in range(10):
        sample_traj.append(ref_kf.sample())
    print(sample_traj)
    

    kf = KalmanFilter(n_obs=2, n_hidden=2, initial_state=ref_kf.inital_state)

    kf.em(sample_traj)


    print(kf.likelihood)