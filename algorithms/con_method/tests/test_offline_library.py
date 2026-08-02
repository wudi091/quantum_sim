import itertools
import unittest

from algorithms.con_method.offline_library import (
    apply_fitted_library_to_episode,
    build_deterministic_scenario_configurations,
    build_waxman_template_pool,
    FixedPathLibraryProblem,
    LibraryScheduleTemplate,
    OfflineLibraryScenario,
    ScenarioConfiguration,
    evaluate_offline_library,
    solve_fixed_path_schedule_library,
    validate_scenario_split,
    waxman_topology_fingerprint,
)
from qnet_core.order_core import OrderBatchProblem, OrderCoreConfig, OrderPlan
from qnet_core.order_core import OrderLinkSpec
from qnet_core.order_waxman import (
    WaxmanOrderConfig,
    WaxmanOrderEpisode,
    WaxmanOrderRequest,
)
from qnet_core.contracts.complete_schedule import enumerate_complete_schedules


def _scenario(
    scenario_id,
    configurations,
    *,
    trace_digest=None,
):
    return OfflineLibraryScenario(
        scenario_id=scenario_id,
        trace_digest=trace_digest or f"trace:{scenario_id}",
        topology_fingerprint="topology:fixed",
        request_distribution_fingerprint="requests:poisson-v1",
        physics_fingerprint="physics:nominal-groups-v1",
        configurations=tuple(configurations),
    )


class OfflineScheduleLibraryTests(unittest.TestCase):
    def test_fitted_library_is_injected_without_core_algorithm_dependency(self):
        path = (0, 1, 2, 3, 4)
        config = WaxmanOrderConfig(
            node_count=5,
            average_degree=2,
            request_count=1,
            arrival_rate=1.0,
            episode_steps=1,
            request_ttl_slots=1,
            min_hops=4,
            max_hops=4,
            candidate_paths=1,
            order_variants_per_path=4,
            node_memory_cap=4,
            swap_probability=1.0,
        )
        episode = WaxmanOrderEpisode(
            seed=0,
            config=config,
            nodes=path,
            links=tuple(
                OrderLinkSpec(left, right, generation_probability=1.0)
                for left, right in zip(path, path[1:])
            ),
            node_capacities=tuple((node, 4) for node in path),
            positions=tuple((node, (float(node), 0.0)) for node in path),
            requests=(WaxmanOrderRequest("r", 0, 4, 0, 1, 4),),
            request_paths=(("r", (path,)),),
            topology_beta=1.0,
            link_alpha=1.0,
            horizon_slots=1,
        )
        pool = build_waxman_template_pool(episode)
        topology = waxman_topology_fingerprint(episode)
        scenario = OfflineLibraryScenario(
            scenario_id="train",
            trace_digest="trace:train",
            topology_fingerprint=topology,
            request_distribution_fingerprint="requests:test",
            physics_fingerprint="physics:test",
            configurations=(
                ScenarioConfiguration("empty"),
                *(
                    ScenarioConfiguration(
                        f"use:{template.template_id}",
                        frozenset({template.template_id}),
                        frozenset({"r"}),
                    )
                    for template in pool.templates
                ),
            ),
        )
        fitted = solve_fixed_path_schedule_library(FixedPathLibraryProblem(
            templates=pool.templates,
            scenarios=(scenario,),
            schedules_per_path=4,
        ))

        installed = apply_fitted_library_to_episode(episode, pool, fitted)
        problem = installed.problem_for_slot(("r",), 0, physics_seed=0)

        self.assertEqual(installed.schedule_library_source, "con-offline-scenario-milp")
        self.assertEqual(
            installed.schedule_library_digest,
            fitted.library.structural_digest,
        )
        self.assertEqual(len(problem.candidates), 4)
        self.assertEqual(
            {plan.schedule_key for plan in problem.candidates},
            {
                pool.template_by_id[template_id].structural_key
                for template_id in fitted.library.selected_template_ids
            },
        )

    def test_configuration_columns_are_validated_by_shared_executor(self):
        problem = OrderBatchProblem.create(
            candidates=(
                OrderPlan("r1", "r1", ("A", "C", "B"), ("C",)),
                OrderPlan("r2", "r2", ("X", "C", "Y"), ("C",)),
            ),
            node_capacity={node: 2 for node in "ACBXY"},
            config=OrderCoreConfig(
                slot_duration_ps=1_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                memory_reset_ps=0,
                generation_probability=1.0,
                swap_probability=1.0,
            ),
        )

        built = build_deterministic_scenario_configurations(
            problem,
            {"r1": "template:r1", "r2": "template:r2"},
        )

        self.assertEqual(built.enumerated_assignments, 4)
        self.assertEqual(built.feasible_assignments, 3)
        self.assertEqual(
            {configuration.completed_request_ids for configuration in built.configurations},
            {frozenset(), frozenset({"r1"}), frozenset({"r2"})},
        )
        self.assertNotIn(
            frozenset({"r1", "r2"}),
            {configuration.completed_request_ids for configuration in built.configurations},
        )

    def test_short_paths_use_effective_budget_without_padding(self):
        paths = {
            "direct": ("S", "T"),
            "one": ("S", "A", "T"),
            "two": ("S", "B", "C", "T"),
            "three": ("S", "D", "E", "F", "T"),
        }
        expected_catalogue_sizes = {
            "direct": 1,
            "one": 1,
            "two": 2,
            "three": 7,
        }
        templates = []
        for path_id, path in paths.items():
            schedules = enumerate_complete_schedules(path)
            self.assertEqual(len(schedules), expected_catalogue_sizes[path_id])
            templates.extend(
                LibraryScheduleTemplate(
                    f"{path_id}:{index}", path_id, schedule
                )
                for index, schedule in enumerate(schedules)
            )

        result = solve_fixed_path_schedule_library(FixedPathLibraryProblem(
            templates=tuple(templates),
            scenarios=(_scenario(
                "train",
                (ScenarioConfiguration("empty"),),
            ),),
            schedules_per_path=4,
        ))

        self.assertEqual(
            result.library.effective_budget_by_path,
            {"direct": 1, "one": 1, "three": 4, "two": 2},
        )
        self.assertEqual(
            {path_id: len(ids) for path_id, ids in result.library.selected_by_path.items()},
            {"direct": 1, "one": 1, "three": 4, "two": 2},
        )
        self.assertEqual(len(result.library.selected_template_ids), 8)
        self.assertEqual(
            enumerate_complete_schedules(paths["direct"])[0].groups,
            (),
        )
        self.assertEqual(
            enumerate_complete_schedules(paths["one"])[0].groups,
            (("A",),),
        )
        self.assertEqual(
            {
                schedule.groups
                for schedule in enumerate_complete_schedules(paths["two"])
            },
            {
                (("B",), ("C",)),
                (("C",), ("B",)),
            },
        )

    @staticmethod
    def _five_by_five_fixture():
        p_schedules = enumerate_complete_schedules(
            ("PS", "P1", "P2", "P3", "PT")
        )[:5]
        q_schedules = enumerate_complete_schedules(
            ("QS", "Q1", "Q2", "Q3", "QT")
        )[:5]
        templates = tuple(
            LibraryScheduleTemplate(f"P:{index}", "P", schedule)
            for index, schedule in enumerate(p_schedules)
        ) + tuple(
            LibraryScheduleTemplate(f"Q:{index}", "Q", schedule)
            for index, schedule in enumerate(q_schedules)
        )
        scenarios = []
        for scenario_index in range(5):
            configurations = [ScenarioConfiguration("empty")]
            configurations.extend(
                ScenarioConfiguration(
                    f"p:{index}",
                    frozenset({f"P:{index}"}),
                    frozenset({f"p-request-{scenario_index}"}),
                )
                for index in range(5)
            )
            configurations.extend(
                ScenarioConfiguration(
                    f"q:{index}",
                    frozenset({f"Q:{index}"}),
                    frozenset({f"q-request-{scenario_index}"}),
                )
                for index in range(5)
            )
            configurations.append(ScenarioConfiguration(
                f"compatible-pair:{scenario_index}",
                frozenset({
                    f"P:{scenario_index}", f"Q:{scenario_index}"
                }),
                frozenset({
                    f"p-request-{scenario_index}",
                    f"q-request-{scenario_index}",
                }),
            ))
            scenarios.append(_scenario(
                f"train-{scenario_index}", configurations
            ))
        return templates, tuple(scenarios)

    @staticmethod
    def _score_library(selected, scenarios):
        total = 0
        for scenario in scenarios:
            total += max(
                configuration.completed_count
                for configuration in scenario.configurations
                if configuration.used_template_ids <= selected
            )
        return total

    def test_offline_milp_matches_independent_nested_bruteforce(self):
        templates, scenarios = self._five_by_five_fixture()
        result = solve_fixed_path_schedule_library(FixedPathLibraryProblem(
            templates=templates,
            scenarios=scenarios,
            schedules_per_path=4,
        ))

        brute_force_best = -1
        brute_force_argmax = []
        p_ids = tuple(f"P:{index}" for index in range(5))
        q_ids = tuple(f"Q:{index}" for index in range(5))
        for selected_p in itertools.combinations(p_ids, 4):
            for selected_q in itertools.combinations(q_ids, 4):
                selected = frozenset(selected_p + selected_q)
                score = self._score_library(selected, scenarios)
                if score > brute_force_best:
                    brute_force_best = score
                    brute_force_argmax = [selected]
                elif score == brute_force_best:
                    brute_force_argmax.append(selected)

        selected = frozenset(result.library.selected_template_ids)
        omitted_p = {f"P:{index}" for index in range(5)} - selected
        omitted_q = {f"Q:{index}" for index in range(5)} - selected
        fixed_different_omissions = frozenset(
            ({f"P:{index}" for index in range(5)} - {"P:0"})
            | ({f"Q:{index}" for index in range(5)} - {"Q:1"})
        )

        self.assertEqual(brute_force_best, 9)
        self.assertEqual(result.training_total_completed, 9)
        self.assertEqual(result.training_weighted_completed, 9)
        self.assertEqual(result.solver_mip_gap, 0.0)
        self.assertIn(selected, brute_force_argmax)
        self.assertEqual(len(omitted_p), 1)
        self.assertEqual(len(omitted_q), 1)
        self.assertEqual(
            next(iter(omitted_p)).split(":")[1],
            next(iter(omitted_q)).split(":")[1],
        )
        self.assertGreaterEqual(
            result.training_total_completed,
            self._score_library(fixed_different_omissions, scenarios),
        )

    def test_training_and_evaluation_are_split_by_trace_content(self):
        schedule = enumerate_complete_schedules(("S", "A", "T"))[0]
        template = LibraryScheduleTemplate("only", "path", schedule)
        train = _scenario(
            "train",
            (
                ScenarioConfiguration("empty"),
                ScenarioConfiguration(
                    "serve",
                    frozenset({"only"}),
                    frozenset({"request"}),
                ),
            ),
            trace_digest="trace:train",
        )
        result = solve_fixed_path_schedule_library(FixedPathLibraryProblem(
            templates=(template,),
            scenarios=(train,),
        ))
        digest_before = result.library.structural_digest
        held_out = _scenario(
            "eval",
            (
                ScenarioConfiguration("empty"),
                ScenarioConfiguration(
                    "serve",
                    frozenset({"only"}),
                    frozenset({"new-request"}),
                ),
            ),
            trace_digest="trace:eval",
        )
        evaluation = evaluate_offline_library(result, (held_out,))

        self.assertEqual(evaluation.total_completed, 1)
        self.assertEqual(evaluation.library_structural_digest, digest_before)
        self.assertEqual(result.library.training_scenario_ids, ("train",))
        self.assertEqual(evaluation.evaluation_scenario_ids, ("eval",))

        leaked = _scenario(
            "different-id",
            (ScenarioConfiguration("empty"),),
            trace_digest="trace:train",
        )
        with self.assertRaisesRegex(ValueError, "trace overlap"):
            validate_scenario_split((train,), (leaked,))


if __name__ == "__main__":
    unittest.main()
