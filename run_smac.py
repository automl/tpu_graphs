from smac import Scenario, HyperparameterOptimizationFacade
from tpu_graphs.baselines.tiles import train_lib

from collections.abc import Sequence

from absl import app


def main(unused_argv: Sequence[str]) -> None:
    scenario = Scenario(
        train_lib.get_config_space(),  # here we pass our search space.
        n_trials=5,  # We want to run max 50 trials (combination of config and seed)
        # deterministic objective function? i.e. do we expect to see noise in
        # our objective for different seeds (which is why we need to spend
        # trials on seeds and average here)
        deterministic=False
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        train_lib.train_model_with_smac_config,  # here we pass our pipeline function that we want to optimize over.
        overwrite=True  # this setting can be ignored for the moment
    )

    incumbent = smac.optimize()
    incumbent_cost = smac.validate(incumbent, seed=1235)
    print(f"Incumbent cost: {incumbent_cost}")
    print(f"Incumbent accuracy {1 - incumbent_cost}")


if __name__ == '__main__':
  app.run(main)
