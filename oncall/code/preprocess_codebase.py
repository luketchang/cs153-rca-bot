import argparse
import os
import tempfile

from common import EXCLUDED_DIRS, clone_repository, collect_files, generate_tree_string
from langchain_core.prompts import ChatPromptTemplate

from oncall.code.models import SelectedModules
from oncall.lib.logging import logger, traceback_log_err
from oncall.lib.utils import get_llm

DIR_SUMMARY_TEMPLATE = """
Your task is to read the following set of codebase files (from a larger directory) and output a module or file level walkthrough of what different modules/files do. For each unit of code you choose, describe what it does and its role in the broader system if applicable.

Additional Instructions
- The final response be resemble a file tree structure with descriptions of each directory/module/file (whatever unit is appropriate) and its purpose/function. If there are many small files with limited functionality/scope, feel free to group them together as a module.
- A reader of the output should be able to read the walkthrough/architecture document and understand how to navigate this part of the repo to find the various modules/components relevant to their task or issue.
- If the files or modules you are looking at are actual services (servers, daemons, etc) or are close in level of abstraction to a running service, provide more extensive detail than you normally would and ensure reponsibilities and functionality are thoroughly described.
- Use markdown for formatting if needed.

System Description: {system_description}

Overall Repository File Tree:
----------------------------------------
{repo_file_tree}
----------------------------------------

Processing Directory: {directory}
Directory File List:
{dir_file_tree}

Source Files (file path and content):
{file_contents}
"""

MERGE_SUMMARIES_TEMPLATE = """
You are tasked with synthesizing a comprehensive system architecture document based on the sub walkthroughs of each major module/directory in the repository. The final document should provide a holistic view of the system, detailing the components, their functions, and interactions. Include a section detailing how the components of the system interact with each other and how the system works as a whole.

Additional Instructions
- The final response be resemble a file tree structure with descriptions of each directory/module/file (whatever unit is appropriate) and its purpose/function. If there are many small files with limited functionality/scope, feel free to group them together as a module.
- A reader of the output should be able to read the walkthrough/architecture document and understand how to navigate the repo to find the various modules/components relevant to their task or issue.
- If the files or modules you are looking at are actual services (servers, daemons, etc) or are close in level of abstraction to a running service, provide more extensive detail than you normally would and ensure reponsibilities and functionality are thoroughly described.
- Use markdown for formatting if needed.

System Description: {system_description}

Overall Repository File Tree:
{repo_file_tree}

Walkthroughs for each major module/directory:
{summaries}

Please provide the final merged system architecture document.
"""

SERVICE_IDENTIFICATION_TEMPLATE = """
You are given the file tree of a repository. Your task is to identify upper-level directories that represent separate services. These directories are independent services that can (or should) be summarized individually.

Return a JSON array of directory paths (relative to the repository root) that should be treated as separate services or packages. Prioritize recall over precision, be generous about including directories as services. But note that you must choose directories not actual files.
Repository File Tree:
{repo_file_tree}
"""


class CodebaseProcessor:
    def __init__(self, llm, repo_url, system_description=""):
        self.llm = llm
        self.repo_url = repo_url
        self.system_description = system_description
        self.allowed_extensions = (
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".rs",
            ".yaml",
            ".cs",
        )

    def list_major_directories(self, repo_path):
        dirs = []
        for item in os.listdir(repo_path):
            full_path = os.path.join(repo_path, item)
            if os.path.isdir(full_path) and not any(
                x in full_path.split(os.sep) for x in EXCLUDED_DIRS
            ):
                dirs.append(full_path)
        return dirs

    def identify_services(self, repo_path, repo_file_tree):
        tree_str = generate_tree_string(repo_file_tree)
        print("File tree:", tree_str)
        prompt_template = ChatPromptTemplate.from_template(
            SERVICE_IDENTIFICATION_TEMPLATE
        )
        chain = prompt_template | self.llm.with_structured_output(SelectedModules)
        structured_output = chain.invoke({"repo_file_tree": tree_str})

        # Log the structured output for debugging
        logger.info(f"Service identification structured output: {structured_output}")

        try:
            # structured_output is expected to be an instance of SelectedModules
            selections = structured_output.selections
            if not selections:
                logger.info(
                    "No service directories identified in structured output; falling back to major directories."
                )
                return []
            abs_service_dirs = []
            for selection in selections:
                rel_path = selection.module
                abs_path = os.path.join(repo_path, rel_path)
                if os.path.exists(abs_path):
                    abs_service_dirs.append(abs_path)
                else:
                    logger.warning(
                        f"Identified service directory '{rel_path}' does not exist in the repository."
                    )
            return abs_service_dirs
        except Exception as e:
            logger.error(
                f"Error processing structured service identification output: {e}"
            )
            return []

    def generate_directory_summary(
        self, directory, dir_file_tree, file_contents, repo_file_tree
    ):
        file_contents_str = ""
        for path, content in file_contents.items():
            file_contents_str += f"\nFile: {path}\n{'-' * 40}\n{content}\n{'-' * 40}\n"
        prompt_template = ChatPromptTemplate.from_template(DIR_SUMMARY_TEMPLATE)
        chain = prompt_template | self.llm
        return chain.invoke(
            {
                "system_description": self.system_description,
                "repo_file_tree": generate_tree_string(repo_file_tree),
                "directory": directory,
                "dir_file_tree": generate_tree_string(dir_file_tree),
                "file_contents": file_contents_str,
            }
        )

    def merge_all_summaries(self, summaries, repo_file_tree):
        summaries_str = ""
        for directory, summary in summaries.items():
            summaries_str += f"Walkthrough for {directory}:\n{summary}\n\n"
        prompt_template = ChatPromptTemplate.from_template(MERGE_SUMMARIES_TEMPLATE)
        chain = prompt_template | self.llm
        return chain.invoke(
            {
                "system_description": self.system_description,
                "repo_file_tree": generate_tree_string(repo_file_tree),
                "summaries": summaries_str,
            }
        )

    def process(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            clone_repository(self.repo_url, tmp_dir)
            repo_path = tmp_dir
            repo_file_tree, _ = collect_files(
                repo_path, self.allowed_extensions, repo_path
            )

            # Use LLM to identify service directories from the full repository tree.
            service_dirs = self.identify_services(repo_path, repo_file_tree)
            if service_dirs:
                directories_to_process = service_dirs
                logger.info(f"Identified service directories: {directories_to_process}")
            else:
                directories_to_process = self.list_major_directories(repo_path)
                logger.info(
                    "No specific service directories identified; falling back to major directories."
                )

            summaries = {}
            for directory in directories_to_process:
                logger.info(f"Processing directory: {directory}")
                dir_file_tree, file_contents = collect_files(
                    directory, self.allowed_extensions, repo_path
                )
                summary = self.generate_directory_summary(
                    directory, dir_file_tree, file_contents, repo_file_tree
                )
                summaries[directory] = summary
                logger.info(f"Summary for {directory}:\n{summary}")

            logger.info("Merging all summaries...")
            final_document = self.merge_all_summaries(summaries, repo_file_tree)
            logger.info(f"Final Document:\n{final_document}")

            return final_document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a system architecture document from a GitHub repository."
    )
    parser.add_argument(
        "--repo_url", required=True, help="GitHub repository URL to clone and analyze"
    )
    parser.add_argument(
        "--model", default="gpt-4o", help="LLM model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--system_description",
        default="",
        help="Optional system description for context",
    )
    args = parser.parse_args()

    try:
        llm = get_llm(model=args.model)
        processor = CodebaseProcessor(
            llm=llm,
            repo_url=args.repo_url,
            system_description=args.system_description,
        )
        final_document = processor.process()
        print("Final Document:\n", final_document)
    except Exception as e:
        traceback_log_err(e)
        traceback_log_err(e)
