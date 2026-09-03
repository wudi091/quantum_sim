from experiments.run_online_experiments import _methods_flags


def test_deployable_method_set_skips_only_milp() -> None:
    assert _methods_flags(["gnn", "qcast", "qpass", "greedy"]) == [
        "--skip-milp"
    ]


def test_method_subset_disables_unselected_baselines() -> None:
    assert _methods_flags(["gnn", "qpass"]) == [
        "--skip-milp",
        "--skip-qcast",
        "--skip-greedy",
    ]
