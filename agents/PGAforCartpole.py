import numpy as np
import inspect
import tensorflow as tf
from capstone import logz
import os
import time
from collections import deque



class PolicyGradientAgent():
    """
    A reinforcement learning agent for the OpenAI gym environment "cartpole-v0".
    It uses policy gradients, based on Geron (2017) - Hands-On Machine Learning with
    Scikit-Learn and TensorFlow
    """
    def __init__(self,
                 environment,
                 learning_rate,
                 discount_rate,
                 exploration_rate,
                 exploration_rate_min,
                 exploration_rate_decay,
                 number_of_episodes_per_update,
                 #replay_sampling_batch_size,
                 nn_architecture,
                 #replay_start_size,
                 exp_name
                 ):
        """
        Initialize the policy gradient agent with the given parameters and build the TensorFlow model.

        :param environment: an OpenAI gym environment
        :param learning_rate: the learning rate used for gradient descent
        :param discount_rate: the discount rate used for reinforcement learning
        :param exploration_rate: the initial exploration rate
        :param exploration_rate_min: the minimum value of the exploration rate achieved after exploration rate decay
        :param exploration_rate_decay: the factor by which the current exploration rate is multiplied per time step
        :param number_of_episode_per_update: how many times is the environment simulated without
                           doing an update of the parametrized policy
        :param nn_architecture: layout of the neural network used for function approximation.
                                [x,y,z] e.g. means that we have 3 hidden layers with x, y and z neurons respectively
        :param exp_name: legend to be shown in result plots
        """

        self.environment = environment
        print("Initializing PG agent...")
        self.state_size = environment.observation_space.shape[0]
        print(" .... dimension of state space:", self.state_size)
        self.action_size = environment.action_space.n
        print(" .... dimension of action space:", self.action_size)

        # define hyperparameters
        self.learning_rate = learning_rate                   # often called alpha
        self.discount_rate = discount_rate                   # often called gamma
        self.exploration_rate = exploration_rate             # often called epsilon
        self.exploration_rate_min = exploration_rate_min
        self.exploration_rate_decay = exploration_rate_decay
        self.number_of_episodes_per_update = number_of_episodes_per_update
        self.nn_architecture = nn_architecture
        self.policyNetwork = self.build_model()

        # Log experimental parameters
        args = inspect.getfullargspec(self.__init__)[0]
        locals_ = locals()
        self.params = {k: locals_[k] if k in locals_ else None for k in args[2:]}
        self.exp_name = exp_name

        # set initial seeds to 0
        tf.set_random_seed(0)
        np.random.seed(0)


    def reset(self):
        """
        resets the exploration_rate to 1.0 and resets the neural network to its original status
        """
        self.policyNetwork = self.build_model()
        self.exploration_rate = 1.0

        # set initial seeds to 0
        tf.set_random_seed(0)
        np.random.seed(0)


    def build_model(self):
        """
        Builds the neural network approximation of the policy with TensorFlow.
        """
        tf.reset_default_graph()

        # specify the NN architecture
        # ------------------------------
        # the number of inputs to the NN is the size of the observation space
        self.numberOfInputs = self.environment.observation_space.shape[0]

        # the output of the NN will be the probability of choosing action==left, so we have
        # exactly one output
        numberOfOutputs = 1

        initializer = tf.contrib.layers.variance_scaling_initializer()

        # build the NN policy approximation
        # ---------------
        # this is just a vanilla multi-layer perceptron
        self.X = tf.placeholder(tf.float32, shape=[None, self.numberOfInputs])

        # depending on self.nn_architecture, build a NN with 1, 2 or 3 hidden layers
        if len(self.nn_architecture) == 1:
            last_hidden = tf.layers.dense(self.X, self.nn_architecture[0], activation=tf.nn.relu,
                                     kernel_initializer=initializer)
        elif len(self.nn_architecture) == 2:
            hidden1 = tf.layers.dense(self.X, self.nn_architecture[0], activation=tf.nn.relu,
                                     kernel_initializer=initializer)
            last_hidden = tf.layers.dense(hidden1, self.nn_architecture[1], activation=tf.nn.relu,
                                     kernel_initializer=initializer)
        elif len(self.nn_architecture) == 3:
            hidden1 = tf.layers.dense(self.X, self.nn_architecture[0], activation=tf.nn.relu,
                                     kernel_initializer=initializer)
            hidden2 = tf.layers.dense(hidden1, self.nn_architecture[1], activation=tf.nn.relu,
                                     kernel_initializer=initializer)
            last_hidden = tf.layers.dense(hidden2, self.nn_architecture[2], activation=tf.nn.relu,
                                     kernel_initializer=initializer)
        else:
            raise ValueError("Wrong NN architecture specified.")


        logits = tf.layers.dense(last_hidden, numberOfOutputs,
                                 kernel_initializer=initializer)

        # use the sigmoid (not the softmax) activation function for the last
        #     layer because we have just 2 possible actions
        outputs = tf.nn.sigmoid(logits)


        ############ action selection ############
        # select random action based on the estimated probabilities
        # --------------------------------------------------------------
        probabilitiesForGoingLeftAndRight = tf.concat(
            axis=1,
            values=[outputs, 1 - outputs]
        )

        # call the multinomial function to pick a random action based on the
        # calculated probabilities
        self.action = tf.multinomial(
            tf.log(probabilitiesForGoingLeftAndRight), num_samples=1)

        ############ define the cost function to train the network ############
        # 4. set the target probability
        #
        # We are acting as though the chosen action is the best possible action
        # which means that the target probability must be 1.0 if the chosen action
        # is 0 (left) and 0.0 if the chosen action is 1 (right)
        y = 1. - tf.to_float(self.action)

        # with the target probability defined, we can now define the cost function
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

        # choose an optimizer, set it's learning rate and compute the gradients of the
        #   the cost function
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # the next command returns a a list of (gradient, variable) pairs where 'gradient' is
        #   the gradient for 'variable'
        gradients_wrt_to_variables = optimizer.compute_gradients(cross_entropy)

        # put gradients into their own list (they will be changed later)
        self.gradients = [gradient for gradient, variable in gradients_wrt_to_variables]

        # during the execution phase, the gradients will be tweaked and reapplied to the
        #  optimizer. first, we need a list that can hold the tweaked gradients. In this list,
        #  we have to initialize some tensorflow placeholders
        self.tweakedGradientPlaceholders = []
        tweakedGradientsAndVariables = []

        for gradient, variable in gradients_wrt_to_variables:
            gradientPlaceholder = tf.placeholder(tf.float32, shape=gradient.get_shape())
            self.tweakedGradientPlaceholders.append(gradientPlaceholder)
            tweakedGradientsAndVariables.append((gradientPlaceholder, variable))

        # for training, we apply these tweaked gradients to the optimizer
        self.trainingOperation = optimizer.apply_gradients(tweakedGradientsAndVariables)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


        #print("Done building TensorFlow graph.")


    def run_numberOfTrials_experiments(
            self,
            numberOfTrials,
            numberOfEpisodesForEachTrial,
            logdir,
            seed=0):
        """
        Perform "numberOfTrials" experiments with different seeds.

        :param numberOfTrials: specifies the number of experiments to perform for this agent
        :param numberOfEpisodesForEachTrial: specifies the maximum number of episodes per experiment
        :param logdir: specifies the logging directory
        :param seed:  the random seed to use for numpy and tensorflow
        """

        allTrials = []

        for trial in range(numberOfTrials):
            # set random seeds
            seed += 1
            tf.set_random_seed(seed)
            np.random.seed(seed)

            # setup new logdir for each experiment
            logz.reset()
            logz.configure_output_dir(os.path.join(logdir, 'trial_{}'.format(trial+1)))
            logz.save_params(self.params)

            thisTrial = self.solve(numberOfEpisodes=numberOfEpisodesForEachTrial,
                                     numberOfTimesteps=500,
                                     trialNumber=trial,  # for logging purposes
                                     seed=seed  # for logging purposes
                                     )

            allTrials.append(thisTrial)


            # reset agent
            self.reset()


        print("Average number of required episodes to solve environment "
              "over {} trials: {}".format(
            numberOfTrials, np.mean(allTrials)))

        filename = 'avg_number_of_required_episodes.txt'
        path_to_file = os.path.join(logdir, filename)
        with open(path_to_file,'w') as file:
            file.write("No. of trials: {}".format(numberOfTrials))
            file.write("      ".format(numberOfTrials))
            file.write("Avg. number of episodes: {}".format(np.mean(allTrials)))


    def solve(self, numberOfEpisodes, numberOfTimesteps, trialNumber, seed):
        """
        "numberOfEpisodes" episodes with a maximum of "numberOfTimesteps" timesteps is performed.

        :param numberOfEpisodes:   how many episodes are to be performed?
        :param numberOfTimesteps:  how many timesteps are to be performed per episode?
        :param trialNumber:        the trial number of the current experiment
                                      (only given to the function for logging  purposes)
        :param seed:               the current random seed (only given to function for
                                      logging purposes)
        :return:                   The number of the episode the environment was solved
                                      (solved is defined as achieving an average reward
                                       of 195 over 100 consecutive episodes)
        """
        print("=====================")
        print("Starting new trial")
        start = time.time()

        lastHundredScores = deque(maxlen=100)

        # declare variable to collect every trajectory
        trajectories = []


        # let the neural network policy play the game several times
        # at each step, compute the gradients that would make the chosen
        # action even more likely, but don't apply these gradients yet

        with tf.Session() as sess:
            self.init.run()

            for episode in range(numberOfEpisodes):
                print("Starting episode {}".format(episode))
                # a list of all non-discounted rewards
                all_states_in_one_episode = []
                all_actions_in_one_episode = []
                all_rewards_in_one_episode = []

                # a list of gradients saved at each step of each episode
                all_gradients_in_one_episode = []


                for simulationEpisode in range(self.number_of_episodes_per_update):
                    # create list for the rewards of the current simulationEpisode
                    rewards_for_emulation_run = []
                    actions_for_emulation_run = []
                    states_for_emulation_run  = []
                    # list of gradients form the current episode
                    gradients_for_emulation_run = []

                    observation = self.environment.reset()

                    for timeStep in range(numberOfTimesteps):
                        states_for_emulation_run.append(observation)
                        all_states_in_one_episode.append(observation)

                        actionValue, gradientValue = sess.run(
                            [self.action, self.gradients],
                            feed_dict={self.X: observation.reshape(1, self.numberOfInputs)}
                        )
                        observation, reward, done, _ = self.environment.step(actionValue[0][0])
                        actions_for_emulation_run.append(actionValue)
                        rewards_for_emulation_run.append(reward)
                        gradients_for_emulation_run.append(gradientValue)
                        if done:
                            lastHundredScores.appendleft(timeStep)
                            break

                    all_actions_in_one_episode.append(actions_for_emulation_run)
                    all_rewards_in_one_episode.append(rewards_for_emulation_run)
                    all_gradients_in_one_episode.append(gradients_for_emulation_run)

                trajectory = {"state" : np.array(states_for_emulation_run),
                              "reward": np.array(rewards_for_emulation_run),
                              "action": np.array(actions_for_emulation_run)
                }
                trajectories.append(trajectory)



                # once we have run the policy for the specified amount of times, it will
                # be updated

                # first, the obtained rewards are discounted
                all_rewards_in_one_episode = self.discount_and_normalize_rewards(
                    all_rewards_in_one_episode, self.discount_rate)
                feed_dict = {}

                for variableIndex, gradientPlaceholder in enumerate(
                        self.tweakedGradientPlaceholders):
                    # multiply the gradients by the action scores and compute the mean
                    meanGradients = np.mean(
                        [reward * all_gradients_in_one_episode[episodeIndex][step][variableIndex]
                         for episodeIndex, rewards in enumerate(all_rewards_in_one_episode)
                         for step, reward in enumerate(rewards)],
                        axis=0
                    )
                    feed_dict[gradientPlaceholder] = meanGradients

                sess.run(self.trainingOperation, feed_dict=feed_dict)

                currentAverageScore = np.mean(lastHundredScores)


                if currentAverageScore >= 195:
                    print("--------------------------------------------------")
                    print("Done. Needed {} episodes to solve the environment.".format(
                        ((episode+1) * self.number_of_episodes_per_update)))
                    print("--------------------------------------------------")
                    break

                returns = [trajectory["reward"].sum() for trajectory in trajectories]
                episode_lengths = [len(trajectory["reward"]) for trajectory in trajectories]
                logz.log_tabular("Experiment", self.exp_name)
                logz.log_tabular("TrialNo", trialNumber + 1)
                logz.log_tabular("seed", seed)
                logz.log_tabular("Episode", ((episode+1) * self.number_of_episodes_per_update))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("AverageReturn", np.mean(returns))
                logz.log_tabular("StdReturn", np.std(returns))
                logz.log_tabular("MaxReturn", np.max(returns))
                logz.log_tabular("MinReturn", np.min(returns))
                logz.log_tabular("AvgScoresFor100Episodes", np.mean(lastHundredScores))
                logz.dump_tabular()


        # return the current episode, which is the episode the environment was solved with
        return (episode + 1) * self.number_of_episodes_per_update 


    # function to compute the total discounted rewards:
    def discount_rewards(self, rewards, discount_rate):
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards

    def discount_and_normalize_rewards(self, all_rewards, discount_rate):
        all_discounted_rewards = [
            self.discount_rewards(rewards, discount_rate) for rewards in all_rewards
        ]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [
            (discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards
        ]


