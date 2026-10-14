from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.build_campaign_manifest import (
    DEFAULT_MANIFEST,
    capture_environment,
    discover_artifacts,
    main,
    render_tcsh_environment,
)


class CampaignManifestTests(unittest.TestCase):
    def load_manifest(self, root: Path) -> dict[str, object]:
        return json.loads((root / DEFAULT_MANIFEST).read_text(encoding="utf-8"))

    def write_manifest(self, root: Path, payload: dict[str, object]) -> None:
        (root / DEFAULT_MANIFEST).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def test_update_preserves_descriptions_and_detects_checksum_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "06_unfolding" / "result.npz"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"frozen-result")
            with mock.patch.dict("os.environ", {"CLAS12ROOT": "/opt/clas12"}, clear=True):
                status = main(
                    [
                        "update",
                        str(root),
                        "--checksum",
                        "all",
                        "--shell-path",
                        "repo=/work/repository",
                    ]
                )
            self.assertEqual(status, 0)
            payload = self.load_manifest(root)
            record = payload["artifacts"][0]
            record["description"] = "Certified unfolding result"
            record["role"] = "unfolding"
            record["tags"] = ["production"]
            record["description_required"] = False
            self.write_manifest(root, payload)

            renamed = artifact.with_name("unfolding_certified.npz")
            artifact.rename(renamed)
            self.assertEqual(
                main(["update", str(root), "--checksum", "changed"]), 0
            )
            updated = self.load_manifest(root)
            present = [
                item for item in updated["artifacts"] if item["status"] == "present"
            ]
            self.assertEqual(len(present), 1)
            self.assertEqual(
                present[0]["path"], "06_unfolding/unfolding_certified.npz"
            )
            self.assertEqual(present[0]["description"], "Certified unfolding result")
            self.assertEqual(present[0]["role"], "unfolding")
            self.assertEqual(present[0]["tags"], ["production"])
            self.assertEqual(present[0]["previous_paths"], ["06_unfolding/result.npz"])
            self.assertEqual(
                updated["environment"]["environment_variables"]["CLAS12ROOT"][
                    "value"
                ],
                "/opt/clas12",
            )
            self.assertEqual(
                updated["environment"]["shell_variables"]["repo"]["value"],
                "/work/repository",
            )

    def test_refresh_retains_missing_records_and_marks_new_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original = root / "03_selection" / "selected.root"
            original.parent.mkdir(parents=True)
            original.write_bytes(b"selected")
            with mock.patch.dict("os.environ", {}, clear=True):
                self.assertEqual(main(["update", str(root)]), 0)
            payload = self.load_manifest(root)
            payload["artifacts"][0]["description"] = "Selected data candidates"
            payload["artifacts"][0]["description_required"] = False
            self.write_manifest(root, payload)

            original.unlink()
            replacement = root / "04_response" / "response.npz"
            replacement.parent.mkdir(parents=True)
            replacement.write_bytes(b"response")
            self.assertEqual(main(["update", str(root)]), 0)
            updated = self.load_manifest(root)
            by_path = {item["path"]: item for item in updated["artifacts"]}
            self.assertEqual(by_path["03_selection/selected.root"]["status"], "missing")
            self.assertEqual(
                by_path["03_selection/selected.root"]["description"],
                "Selected data candidates",
            )
            self.assertEqual(by_path["04_response/response.npz"]["status"], "present")
            self.assertTrue(
                by_path["04_response/response.npz"]["description_required"]
            )
            by_path["04_response/response.npz"]["description"] = "Response matrix"
            by_path["04_response/response.npz"]["description_required"] = False
            self.write_manifest(root, updated)
            self.assertEqual(main(["check", str(root)]), 1)
            self.assertEqual(main(["check", str(root), "--allow-missing"]), 0)

    def test_check_detects_changes_and_requires_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "artifact.txt"
            artifact.write_text("first", encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                self.assertEqual(main(["update", str(root), "--checksum", "all"]), 0)
            self.assertEqual(main(["check", str(root)]), 1)

            payload = self.load_manifest(root)
            payload["artifacts"][0]["description"] = "Small test artifact"
            payload["artifacts"][0]["description_required"] = False
            self.write_manifest(root, payload)
            self.assertEqual(main(["check", str(root), "--verify-checksums"]), 0)

            artifact.write_text("changed and longer", encoding="utf-8")
            self.assertEqual(main(["check", str(root)]), 1)

    def test_environment_restore_handles_unset_and_shell_local_variables(self) -> None:
        snapshot = capture_environment(
            {
                "PATH": "/bin:/opt/tools",
                "RCDB_CONNECTION": "mysql://rcdb@clasdb/rcdb",
                "API_TOKEN": "do-not-store",
            },
            names=("PATH", "RCDB_CONNECTION", "QADB", "API_TOKEN"),
            prefixes=(),
            shell_variables=(
                ("repo", "/work/repo with space", "path"),
                ("label", 'RGK $test "quoted"!', "value"),
            ),
        )
        self.assertIn("API_TOKEN", snapshot["omitted_sensitive_variables"])
        self.assertNotIn("API_TOKEN", snapshot["environment_variables"])
        script = render_tcsh_environment(snapshot)
        self.assertIn("setenv PATH '/bin:/opt/tools'", script)
        self.assertIn("unsetenv QADB", script)
        self.assertIn("set repo = '/work/repo with space'", script)
        self.assertIn("set label = 'RGK $test \"quoted\"\\!'", script)

    @unittest.skipUnless(shutil.which("tcsh"), "tcsh is required")
    def test_generated_environment_can_be_sourced_by_tcsh(self) -> None:
        snapshot = capture_environment(
            {"PATH": "/bin:/usr/bin"},
            names=("PATH", "QADB"),
            prefixes=(),
            shell_variables=(
                ("repo", "/work/repo with space", "path"),
                ("label", 'RGK $test "quoted"!', "value"),
                ("note", "Andrey's result", "value"),
                ("literal", r"a\b", "value"),
            ),
        )
        script_text = render_tcsh_environment(snapshot)
        with tempfile.TemporaryDirectory() as temporary_directory:
            script_path = Path(temporary_directory) / "restore.tcsh"
            script_path.write_text(script_text, encoding="utf-8")
            command = (
                f'source "{script_path}"; '
                'setenv RESTORED "$repo|$label|$note|$literal|$PATH"; '
                "/usr/bin/printenv RESTORED"
            )
            result = subprocess.run(
                ["/bin/tcsh", "-f", "-c", command],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, script_text + result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            '/work/repo with space|RGK $test "quoted"!|Andrey\'s result|a\\b|/bin:/usr/bin',
        )

    def test_discovery_records_symlink_without_traversing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outside = root.parent / f"{root.name}-outside"
            outside.mkdir()
            try:
                (outside / "external.root").touch()
                (root / "external").symlink_to(outside, target_is_directory=True)
                records = discover_artifacts(root, set())
                self.assertEqual([record["path"] for record in records], ["external"])
                self.assertEqual(records[0]["type"], "symlink")
            finally:
                (outside / "external.root").unlink()
                outside.rmdir()


if __name__ == "__main__":
    unittest.main()
