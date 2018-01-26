import gym

def importCartpole():
    """
    Initializes the OpenAI cartpole-v0 environment
    :return: the OpenAI cartpole-v0 environment
    """
    # import the cartpole environment
    envString = "CartPole-v0"
    env = gym.make(envString)

    print()
    print("Importing environment", envString)
    print("----------------------------------")
    print(envString+"'s action space:      ", env.action_space)
    print(envString+"'s observation space: ", env.observation_space)

    print()
    print("For the cartpole environment, the observation space is: ")
    print("obs[0]: the horizontal position of the cart (0.0 = center)")
    print("obs[1]: the velocity of the cart (0.0 = standing still)")
    print("obs[2]: the angle of the pole (0.0 = vertical)")
    print("obs[3]: the angular velocity of the cartpole (0.0 = standing still)")
    return env

