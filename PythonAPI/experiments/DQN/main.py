from dqn import DQN, Agent, ReplayBuffer
from world import World


def test_world():
    world = World()
    world.step(0)
    world.destroy_all()


def test_agent():
    world = World()
    memory = ReplayBuffer(200)
    net = DQN(55, 3)
    agent = Agent(world, memory)
    action = agent.get_action(net, 1, "gpu")

    world.step(0)
    print(action)
    world.destroy_all()


if __name__ == "__main__":
    test_agent()
