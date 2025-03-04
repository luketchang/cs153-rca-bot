import argparse
import os
import tempfile
from typing import Mapping

from langchain_core.prompts import ChatPromptTemplate

from oncall.code.common import (
    clone_repository,
    collect_files,
    generate_tree_string,
    print_source_files,
)
from oncall.code.models import SelectedModules
from oncall.lib.logging import logger, traceback_log_err
from oncall.lib.utils import get_llm

SYSTEM_PROMPT = "Your are an expert software engineer with experience debugging production issues in large codebases for telemetry data and source code. In particular, you are an expert at receiving a system alert or customer support ticket and identifying which services are necessary to investigate to understand the issue and triage it."

INSTRUCTION_PROMPT = """
Given the following codebase/system walkthrough, the codebase file tree, and an issue report (either from system alert or customer support ticket), identify the top 5 upper level services to use as context for triaging the issue in production.
- Given those services/modules, we will manually bring in any imports to complete the picture.
- Prioritize completeness, even if a given service might not be the exact source of the issue if you want that service's code and logs for more complete understanding/diagnosis, include that service.
- Prioritize services (e.g. servers, daemons, etc) over libraries/modules, as libraries/modules will be imported by services. DO NOT include libraries/modules unless you really feel all possibly useful services for diagnosis have already been chosen.
- We are optimizing here for good recall such that we have chosen the correct general services to look at and have enough context when debugging a production issue given that source code + logs/metrics/traces.
- SelectedModules the sources as a list of services and reason for including.
- Provide file paths to services/modules not individual files themselves.
- Use the provided file tree below to know what the actual full paths are and just copy over the names, do not add any additional prefixes or suffixes to the path.
- If there is an infrastructure directory (cloud infrastructure configuration files), include that as a service and pick it.

Codebase/system walkthrough:
{walkthrough}

File tree:
{file_tree}

Production issue:
{issue}

Now, identify the top 5 upper level services to use as context for triaging the issue in production.
"""


class ModuleSelector:
    def __init__(self, llm=get_llm("gpt-4o")):
        self.llm = llm
        self.allowed_extensions = (".py", ".js", ".ts", ".java", ".go", ".rs", ".yaml")

    def select_module_paths(self, walkthrough, file_tree, issue) -> SelectedModules:
        prompt = ChatPromptTemplate(
            [
                ("system", SYSTEM_PROMPT),
                ("human", INSTRUCTION_PROMPT),
            ]
        )
        chain = prompt | self.llm.with_structured_output(SelectedModules)
        return chain.invoke(
            {"walkthrough": walkthrough, "file_tree": file_tree, "issue": issue}
        )

    def expand_module_sources(self, selected_dirs, repo_root) -> Mapping[str, str]:
        all_paths_to_source_code = {}
        for selected_dir in selected_dirs:
            dir_path = os.path.join(repo_root, selected_dir)
            if os.path.isdir(dir_path):
                # collect_files returns a mapping of relative file path -> source code.
                _, paths_to_source_code = collect_files(
                    dir_path, self.allowed_extensions, repo_root
                )
                all_paths_to_source_code.update(paths_to_source_code)
            else:
                logger.warning(f"{dir_path} is not a directory or does not exist.")
        return all_paths_to_source_code

    def select(self, walkthrough, file_tree, issue, repo_root) -> Mapping[str, str]:
        output = self.select_module_paths(walkthrough, file_tree, issue)
        selected_dirs = [selection.module for selection in output.selections]
        return self.expand_module_sources(selected_dirs, repo_root)


def main():
    parser = argparse.ArgumentParser(
        description="Select top 5 upper level services for triaging a production issue."
    )
    parser.add_argument(
        "--repo_url",
        required=True,
        help="Git repository URL to clone and analyze (file tree will be generated from this repo).",
    )
    parser.add_argument(
        "--walkthrough_path", required=True, help="Path to the walkthrough txt file."
    )
    parser.add_argument(
        "--issue",
        required=True,
        help="Production issue description (e.g. alert or support ticket details).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model to use (default: gpt-4o).",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Specific branch to clone (default is master/main if not provided).",
    )
    args = parser.parse_args()

    try:
        with open(args.walkthrough_path, "r", encoding="utf-8") as f:
            walkthrough = f.read()

        logger.info("Using walkthrough:\n" + walkthrough)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Clone the repository into a temporary directory with the specified branch if provided.
            clone_repository(args.repo_url, tmp_dir, branch=args.branch)

            # Define allowed file extensions for code files
            allowed_extensions = (".py", ".js", ".ts", ".java", ".go", ".rs", ".yaml")

            # Collect the file tree (list of file paths) from the repo
            repo_file_tree, _ = collect_files(
                tmp_dir, allowed_extensions, repo_root=tmp_dir
            )
            file_tree_str = generate_tree_string(repo_file_tree)
            logger.info(
                "File tree generated successfully. File tree:\n" + file_tree_str
            )

            # Initialize the ModuleSelector with the provided model
            selector = ModuleSelector(llm=get_llm(args.model))
            modules = selector.select_module_paths(
                walkthrough, file_tree_str, args.issue
            )
            print("Selected Modules:\n", modules)

            selected_dirs = [selection.module for selection in modules.selections]
            sources_code = selector.expand_module_sources(selected_dirs, tmp_dir)
            print_source_files(sources_code)

    except Exception as e:
        traceback_log_err(e)


if __name__ == "__main__":
    main()
