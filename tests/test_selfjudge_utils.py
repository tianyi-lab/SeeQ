"""Exercise the real judge flow with deterministic inference and no model downloads."""

import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    """Load a source module while its optional runtime dependencies are stubbed."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SelfJudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only inference and unused tabular/progress dependencies are replaced;
        # evaluation merging and all judge logic run from the production source.
        qwen = types.ModuleType('gen.scripts.qwen_unified')
        qwen.model_infer_qwen = Mock()
        tqdm = types.ModuleType('tqdm')
        tqdm.tqdm = lambda items, **kwargs: items
        with patch.dict(sys.modules, {
            'gen.scripts.qwen_unified': qwen,
            'pandas': types.ModuleType('pandas'),
            'tqdm': tqdm,
        }):
            analysis = load_module('utils.gen.analysis_utils', 'utils/gen/analysis_utils.py')
            with patch.dict(sys.modules, {'utils.gen.analysis_utils': analysis}):
                cls.judge = load_module('utils.gen.selfjudge_utils', 'utils/gen/selfjudge_utils.py')
        cls.analysis = analysis

    def setUp(self):
        self.hard = {'hard_constraints': {
            'Factuality': {'count': 0},
            'Ambiguity': {'count': 0},
            'Validity': {'answerable': True},
        }}
        self.image = object()
        self.config = {'model': object()}
        self.question = 'Which object is left of the chair?'

    def run_judge(self, responses, question=None, answer='table', constraints=None):
        infer = Mock(side_effect=[(response, '') for response in responses])
        with patch.object(self.judge, 'model_infer_qwen', infer):
            result = self.judge.self_judge(
                self.question if question is None else question,
                answer, self.image, self.config, constraints=constraints,
            )
        return result, infer

    def test_valid_question_runs_hard_gate_before_difficulty(self):
        perception = {'soft_constraints': {'perception_difficulty': {
            'image': {'easy_count': 1, 'hard_count': 2},
        }}}
        reasoning = {'soft_constraints': {'reasoning_difficulty': {'count': 3}}}
        (accepted, result), infer = self.run_judge([self.hard, 'table, chair', perception, reasoning])
        self.assertTrue(accepted)
        self.assertEqual(infer.call_count, 4)
        first = infer.call_args_list[0].kwargs
        self.assertIs(first['dict_input']['image'], self.image)
        self.assertIs(first['dict_config_m'], self.config)
        self.assertEqual(first['dict_input']['prompt'], self.judge.hard_constraints() + 'Question: ' + self.question)
        self.assertEqual(result['evaluation'], [self.hard, perception, reasoning, {}])
        merged = self.analysis._merge_api_res_from_evaluation(result['evaluation'])
        self.assertEqual(merged['hard_constraints'], self.hard['hard_constraints'])
        self.assertEqual(merged['soft_constraints']['perception_difficulty']['image']['count'], 5)
        self.assertEqual(merged['soft_constraints']['reasoning_difficulty']['count'], 3)

    def test_each_hard_failure_stops_inference_and_keeps_evidence(self):
        for section, key, value in [('Factuality', 'count', 1), ('Ambiguity', 'count', 1), ('Validity', 'answerable', False)]:
            with self.subTest(section=section):
                hard = copy.deepcopy(self.hard)
                hard['hard_constraints'][section][key] = value
                (accepted, result), infer = self.run_judge([hard])
                self.assertFalse(accepted)
                self.assertEqual(infer.call_count, 1)
                self.assertEqual(result['evaluation'], [hard])

    def test_malformed_hard_responses_fail_closed(self):
        for response in [None, 'not JSON', [], {}, {'hard_constraints': None}, {'hard_constraints': []},
                         {'hard_constraints': {'Factuality': None}},
                         {'hard_constraints': {'Factuality': [], 'Ambiguity': {}, 'Validity': {}}}]:
            with self.subTest(response=response):
                (accepted, _), infer = self.run_judge([response])
                self.assertFalse(accepted)
                self.assertEqual(infer.call_count, 1)
                self.assertFalse(self.judge.is_valid_hard_constraints(response))

    def test_answerability_must_be_boolean_true(self):
        for value in ['true', 1, None]:
            with self.subTest(value=value):
                self.hard['hard_constraints']['Validity']['answerable'] = value
                self.assertFalse(self.judge.is_valid_hard_constraints(self.hard))

    def test_cheap_rejections_do_not_run_inference(self):
        for question, answer in [('What is it? Where is it?', 'table'), (self.question, None)]:
            (accepted, _), infer = self.run_judge([], question=question, answer=answer)
            self.assertFalse(accepted)
            infer.assert_not_called()
        with patch.object(self.judge, 'model_infer_qwen') as infer:
            self.assertEqual(self.judge.self_judge(None, 'table', self.image, self.config), (False, {}))
            infer.assert_not_called()

    def test_soft_parse_failure_preserves_hard_diagnostics(self):
        for responses in [[self.hard, 'entities', 'bad JSON'], [self.hard, 'entities', {}, 'bad JSON']]:
            with self.subTest(responses=responses):
                (accepted, result), _ = self.run_judge(responses)
                self.assertFalse(accepted)
                self.assertEqual(result['evaluation'], [self.hard])

    def test_constraint_following_remains_available(self):
        constraints = ['Compare positions']
        following = {'soft_constraints': {'model_following_capability': {'by_constraint': [
            {'constraint_text': constraints[0], 'complies': True},
        ]}}}
        (accepted, result), infer = self.run_judge([self.hard, 'entities', {}, {}, following], constraints=constraints)
        self.assertTrue(accepted)
        self.assertEqual(infer.call_count, 5)
        self.assertEqual(result['question_info']['constraint'], constraints)
        self.assertTrue(self.judge.has_required_model_following(result, constraints))


if __name__ == '__main__':
    unittest.main()
