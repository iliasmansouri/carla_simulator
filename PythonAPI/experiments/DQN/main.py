from dqn import DQN, Agent, ReplayBuffer, DQNLightning
from world import World
import pytorch_lightning as pl


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


def test_dqn():
    world = World()
    model = DQNLightning(world)

    trainer = pl.Trainer(gpus=1, fast_dev_run=True)
    trainer.fit(model)

    world.destroy_all()


if __name__ == "__main__":
    # test_agent()
    test_dqn()