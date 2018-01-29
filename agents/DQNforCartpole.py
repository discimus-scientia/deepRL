import numpy as np
import random
import tensorflow as tf
import keras
from collections import deque
from util import logz
import inspect
import time
import os

class DQNforCartpole:
    """
    A reinforcement learning agent for the OpenAI gym environment "cartpole-v0".
    It uses deep Q learning, as it was proposed by Mnih et al. (2013)
    TODO: full paper citation
    """
    def __init__(self,
                 environment,
                 learning_rate,
                 discount_rate,
                 exploration_rate,
                 exploration_rate_min,
                 exploration_rate_decay,
                 replay_memory_capacity,
                 replay_sampling_batch_size,
                 nn_architecture,
                 replay_start_size,
                 exp_name
                 ):
        """
        Initialize the Deep Q learning agent with the given parameters and build the TensorFlow model.

        :param environment: an OpenAI gym environment
        :param learning_rate: the learning rate used for gradient descent
        :param discount_rate: the discount rate used for reinforcement learning
        :param exploration_rate: the initial exploration rate
        :param exploration_rate_min: the minimum value of the exploration rate achieved after exploration rate decay
        :param exploration_rate_decay: the factor by which the current exploration rate is multiplied per time step
        :param replay_memory_capacity: size of the replay memory
        :param replay_sampling_batch_size: how many batches of experiences to sample?
        :param nn_architecture: layout of the neural network used for function approximation.
                                [x,y,z] e.g. means that we have 3 hidden layers with x, y and z neurons respectively
        :param replay_start_size: how many timesteps are performed before the agents starts remembering
        :param exp_name: legend to be shown in result plots
        """

        self.environment = environment
        print("Initializing DQN agent...")
        self.state_size = environment.observation_space.shape[0]
        print(" .... dimension of state space:", self.state_size)
        self.action_size = environment.action_space.n
        print(" .... dimension of action space:", self.action_size)

        # for the replay memory
        self.replay_memory_capacity = replay_memory_capacity
        self.memory = deque(maxlen=self.replay_memory_capacity)     # values taken from keon.io

        # define hyperparameters
        self.learning_rate = learning_rate                   # often called alpha
        self.discount_rate = discount_rate                   # often called gamma
        self.exploration_rate = exploration_rate             # often called epsilon
        self.exploration_rate_min = exploration_rate_min
        self.exploration_rate_decay = exploration_rate_decay
        self.replay_start_size = replay_start_size
        self.replay_sample_batch_size = replay_sampling_batch_size
        self.nn_architecture = nn_architecture
        self.QNetwork = self.build_model()

        # Log experimental parameters
        args = inspect.getfullargspec(self.__init__)[0]
        locals_ = locals()
        self.params = {k: locals_[k] if k in locals_ else None for k in args[2:]}
        self.exp_name = exp_name

    def reset(self):
        """
        resets the exploration_rate to 1.0 and resets the neural network to its original status
        """
        self.QNetwork = self.build_model()
        self.exploration_rate = 1.0

    def build_model(self):
        """
        Builds the neural network model depending on the parameters used to initialize the DQNforCartpole
        instance with keras.
        :return:  a keras model
        """
        model = keras.models.Sequential()

        # depending on the parameters used to initialize the agent, build the network with 1, 2 or 3 hidden
        # layers
        if len(self.nn_architecture) == 1:
            model.add(keras.layers.Dense(self.nn_architecture[0], input_dim=self.state_size, activation='relu'))
        elif len(self.nn_architecture) == 2:
            model.add(keras.layers.Dense(self.nn_architecture[0], input_dim=self.state_size, activation='relu'))
            model.add(keras.layers.Dense(self.nn_architecture[1], activation='relu'))
        elif len(self.nn_architecture) == 3:
            model.add(keras.layers.Dense(self.nn_architecture[0], input_dim=self.state_size, activation='relu'))
            model.add(keras.layers.Dense(self.nn_architecture[1], activation='relu'))
            model.add(keras.layers.Dense(self.nn_architecture[2], activation='relu'))
        else:
            raise ValueError("wrong dimension of nn_architecture")

        # add the output layer with linear activation function
        model.add(keras.layers.Dense(self.action_size, activation='linear'))

        # compile the model with the MSE as loss and the Adam optimizer
        model.compile(loss="mse",
                      optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        """
        The agent chooses an action based on the current state.

        :param state: the state of the environment
        :return: either a randomly chosen action or the argmax of the Q-network
        """

        # use epsilon-greedy exploration strategy: if a random number between 0 and 1
        #    is smaller equal the exploration_rate hyperparameter, a randomly chosen action
        #    is returned.
        if np.random.rand() <= self.exploration_rate:
            return self.environment.action_space.sample()

        # get the q-values for the current state
        all_q_values = self.QNetwork.predict(state)

        # return the action with the highest q-value
        return np.argmax(all_q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """
        Append an experience to the replay memory
        :param state: the current state
        :param action:  the action taken in the current state
        :param reward:  the reward obtained for choosing action "action" in state "state"
        :param next_state: the next state
        :param done: flag that determines whether an episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Replays a remembered experience.

        First, a random sample of experiences with size "batch_size" is chosen from the replay memory. Then,
        the loss function is defined, based on the experience sampled. TODO: explain more about it

        :param batch_size: How many episodes should be sampled from the replay memory?
        """
        random_experience = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in random_experience:
            # TODO: explain all this in more detail
            target = reward
            if not done:
                target = reward + self.discount_rate * np.amax(
                    self.QNetwork.predict(next_state)[0])

            target_f = self.QNetwork.predict(state)
            target_f[0][action] = target
            self.QNetwork.fit(state, target_f, epochs=1, verbose=0)

        # decay the exploration rate
        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_rate_decay

    def solve(self, numberOfEpisodes=5000, numberOfTimesteps=500, trialNumber=None, seed=0):
        """
        "numberOfEpisodes" episodes with a maximum of "numberOfTimesteps" timesteps is performed.

        :param numberOfEpisodes:   how many episodes are to be performed?
        :param numberOfTimesteps:  how many timesteps are to be performed per episode?
        :param trialNumber:        the trial number of the current experiment (only given to the function for logging
                                       purposes)
        :param seed:               the current random seed (only given to function for logging purposes)

        :return: The number of the episode the environment was solved (solved is defined as achieving an average reward
                        of 195 over 100 consecutive episodes
        """
        # first, add the number of episodes to the replay_start_size, so that the loop is done the correct number of
        #     times
        numberOfEpisodes += self.replay_start_size
        print("=====================")
        print("Starting new trial")
        start = time.time()

        lastHundredScores = deque(maxlen=100)

        # declare variable to collect every trajectory
        trajectories = []

        for episode in range(numberOfEpisodes):
            state = self.environment.reset()
            state = np.reshape(state, [1, 4])
            all_states_in_one_episode, all_actions_in_one_episode, all_rewards_in_one_episode = [], [], []

            for timeStep in range(numberOfTimesteps):
                #env.render()
                all_states_in_one_episode.append(state)

                # determine next action to take
                action = self.act(state=state)
                all_actions_in_one_episode.append(action)

                # perform action
                next_state, reward, done, _ = self.environment.step(action)

                all_rewards_in_one_episode.append(reward)
                next_state = np.reshape(next_state, [1, 4])

                # remember this transition
                self.remember(state=state,
                              action=action,
                              reward=reward,
                              next_state=next_state,
                              done=done)

                # make next state the current state
                state = next_state

                if done:
                    # append final score for this episode to lastHundredScores and
                    # break the loop
                    lastHundredScores.appendleft(timeStep)
                    break

            trajectory = {"state" : np.array(all_states_in_one_episode),
                          "reward": np.array(all_rewards_in_one_episode),
                          "action": np.array(all_actions_in_one_episode)
                          }

            trajectories.append(trajectory)

            # specify replay start size
            #    if we have not yet had replay_start_size number of episodes, do not learn anything, yet
            #    instead, start afresh with the game emulation and record the next trajectory
            if episode < self.replay_start_size:
                if ((episode + 1) % 100 == 0):
                    print("We have not reached the specified number of episodes to start replaying, yet")
                    print("Episode {} / {} ".format(episode+1, self.replay_start_size))
                continue

            # train the Q-network with the trajectory of this episode
            self.replay(batch_size=self.replay_sample_batch_size)

            currentAverageScore = np.mean(lastHundredScores)

            # verbose output
            if (episode+1) % 100 == 0:
                print("---------------------")
                print("episode: {}/{}, score for this episode: {}".format((episode - self.replay_start_size)+1,
                                                                          numberOfEpisodes - self.replay_start_size,
                                                                          timeStep))
                print("current epsilon: {}".format(self.exploration_rate))
                print("current average score: {}".format(currentAverageScore))

            if currentAverageScore >= 195:
                print("--------------------------------------------------")
                print("Done. Needed {} episodes to solve the environment.".format((episode - self.replay_start_size)+1))
                print("--------------------------------------------------")
                break

            returns = [trajectory["reward"].sum() for trajectory in trajectories]
            episode_lengths = [len(trajectory["reward"]) for trajectory in trajectories]
            logz.log_tabular("TrialNo", trialNumber + 1)
            logz.log_tabular("seed", seed)
            logz.log_tabular("Episode", (episode - self.replay_start_size) + 1)
            logz.log_tabular("Time", time.time() - start)
            logz.log_tabular("AverageReturn", np.mean(returns))
            logz.log_tabular("StdReturn", np.std(returns))
            logz.log_tabular("MaxReturn", np.max(returns))
            logz.log_tabular("MinReturn", np.min(returns))
            logz.log_tabular("Epsilon", self.exploration_rate)
            logz.log_tabular("EpLenMean", np.mean(episode_lengths))
            logz.log_tabular("EpLenStd", np.std(episode_lengths))
            logz.log_tabular("AvgScoresFor100Episodes", np.mean(lastHundredScores))
            logz.dump_tabular()

        # return the current episode, which is the episode the environment was solved with
        return episode - self.replay_start_size + 1

    def run_numberOfTrials_experiments(
            self,
            numberOfTrials=2,
            numberOfEpisodesForEachTrial=3000,
            logdir=None,
            seed=0):
        """
        Perform "numberOfTrials" experiments with different seeds.

        :param numberOfTrials:
        :param numberOfEpisodesForEachTrial:
        :param logdir:
        :param seed:
        :return:
        """

        allEpisodes = []

        for trial in range(numberOfTrials):

            # set random seeds
            seed += 1
            tf.set_random_seed(seed)
            np.random.seed(seed)

            # setup new logdir for each experiment
            logz.reset()
            logz.configure_output_dir(os.path.join(logdir, 'trial_{}'.format(trial+1)))
            logz.save_params(self.params)

            thisEpisode = self.solve(numberOfEpisodes=numberOfEpisodesForEachTrial,
                                     numberOfTimesteps=500,
                                     trialNumber=trial,  # for logging purposes
                                     seed=seed  # for logging purposes
                                     )

            allEpisodes.append(thisEpisode)

            # reset agent
            self.reset()

        print("Average number of required episodes to solve environment "
              "over {} trials: {}".format(
            numberOfTrials, np.mean(allEpisodes)))

        filename = 'avg_number_of_required_episodes.txt'
        path_to_file = os.path.join(logdir, filename)
        with open(path_to_file,'w') as file:
            file.write("No. of trials: {}".format(numberOfTrials))
            file.write("      ".format(numberOfTrials))
            file.write("Avg. number of episodes: {}".format(np.mean(allEpisodes)))

