from stable_baselines3.common.callbacks import BaseCallback
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay

MODEL_SAVE_NAME = "callback_model"


class DQNCustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(DQNCustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.episodes = 0
        self.total_episode_reward = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        # update commulative reward to log at the end of every episode
        self.total_episode_reward += self.locals["reward"]
        # at the end of every episode
        if self.locals["done"][0]:
            # if log interval has passed
            if self.episodes % self.locals["log_interval"] == 0:
                # Print the last video
                # Save your model and optimizer
                self.model.save(MODEL_SAVE_NAME)
                mp4list = glob.glob("video/*.mp4")
                print(mp4list)
                if len(mp4list) > 0:
                    print(len(mp4list))
                    mp4 = mp4list[-1]
                    video = io.open(mp4, "r+b").read()
                    encoded = base64.b64encode(video)

                    # display gameplay video
                    ipythondisplay.clear_output(wait=True)
                    ipythondisplay.display(
                        HTML(
                            data="""<video alt="" autoplay 
                                    loop controls style="height: 400px;">
                                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                                </video>""".format(
                                encoded.decode("ascii")
                            )
                        )
                    )
                    print("Episode:", self.episodes)
            self.episodes += 1
            self.total_episode_reward = 0

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
