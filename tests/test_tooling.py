"""Offline tests using original synthetic fixtures, without domain datasets."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from de_bench.cli import main
from de_bench.loader import load_tasks
from de_bench.models import Domain, TaskResult
from de_bench.reporting import term_summary


def fixture(identifier='fixture', domain='hu'):
    return dict(id=identifier, domain=domain, category='synthetic', prompt='Name alpha.',
                reference='alpha', rubric=['alpha', 'beta'], scoring='keyword_presence')


class ToolingTests(unittest.TestCase):
    def test_load_file_and_filter_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'one.json').write_text(json.dumps(fixture()))
            (root / 'nested').mkdir()
            (root / 'nested/two.json').write_text(json.dumps([fixture('second', 'legal')]))
            self.assertEqual(len(load_tasks(source=root)), 2)
            self.assertEqual(len(load_tasks(Domain.HU, root)), 1)
            self.assertEqual(load_tasks(source=root / 'one.json')[0].id, 'fixture')

    def test_unverified_and_duplicate_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'tasks.json'
            p.write_text(json.dumps([dict(fixture(), verified=False),
                                     dict(fixture(), metadata={'verified': False})]))
            self.assertEqual(load_tasks(source=p), [])
            p.write_text(json.dumps([fixture(), fixture()]))
            with self.assertRaisesRegex(ValueError, 'Duplicate task ID'):
                load_tasks(source=p)

    def test_invalid_file_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'tasks.json'
            p.write_text('{broken')
            result = CliRunner().invoke(main, ['list', '--tasks', str(p)])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn('Invalid task file', result.output)

    def test_cli_requires_user_data(self):
        result = CliRunner().invoke(main, ['run', '-m', 'ollama/test'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('--tasks', result.output)

    def test_run_and_compare_local_outputs(self):
        class FakeRunner:
            model_name = 'local/test'

            def run(self, task):
                return TaskResult(task.id, self.model_name, 'alpha')

        with tempfile.TemporaryDirectory() as tmp, patch('de_bench.cli.get_runner', return_value=FakeRunner()), patch('de_bench.cli.resource_hashes', return_value={'synonyms': None, 'decompound': None}):
            root = Path(tmp)
            p = root / 'tasks.json'
            p.write_text(json.dumps(fixture()))
            output = root / 'nested/run.json'
            result = CliRunner().invoke(main, ['run', '--tasks', str(p), '-m', 'ollama/test', '-o', str(output)])
            self.assertEqual(result.exit_code, 0, result.output)
            saved = json.loads(output.read_text())
            self.assertEqual(saved['summary']['term_coverage'], 0.5)
            self.assertEqual(saved['metadata']['tasks'][0]['id'], 'fixture')
            self.assertIsNone(saved['metadata']['dictionary_sha256']['synonyms'])
            self.assertIn('Term coverage', result.output)
            other = root / 'other.json'
            other.write_text(output.read_text())
            result = CliRunner().invoke(main, ['compare', str(output), str(other)])
            self.assertEqual(result.exit_code, 0, result.output)
            saved['results'][0]['task_id'] = 'different'
            other.write_text(json.dumps(saved))
            result = CliRunner().invoke(main, ['compare', str(output), str(other)])
            self.assertNotEqual(result.exit_code, 0)

    def test_empty_tasks_do_not_call_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch('de_bench.cli.get_runner') as runner:
            p = Path(tmp) / 'tasks.json'
            p.write_text('[]')
            result = CliRunner().invoke(main, ['run', '--tasks', str(p), '-m', 'ollama/test'])
            self.assertNotEqual(result.exit_code, 0)
            runner.assert_not_called()

    def test_missing_dictionary_fails_before_model_call(self):
        with tempfile.TemporaryDirectory() as tmp, patch('de_bench.cli.resource_hashes', side_effect=FileNotFoundError('missing')), patch('de_bench.cli.get_runner') as runner:
            p = Path(tmp) / 'tasks.json'
            p.write_text(json.dumps(fixture()))
            result = CliRunner().invoke(main, ['run', '--tasks', str(p), '-m', 'ollama/test'])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn('Cannot read configured dictionary', result.output)
            runner.assert_not_called()

    def test_pooled_coverage(self):
        results = [TaskResult('a', 'm', '', details={'hits': ['alpha'], 'misses': []}),
                   TaskResult('b', 'm', '', details={'hits': [], 'misses': ['beta', 'gamma', 'delta']})]
        self.assertEqual(term_summary(results)['term_coverage'], 0.25)
        self.assertIsNone(term_summary([])['term_coverage'])


if __name__ == '__main__':
    unittest.main()
