import json
import os
import subprocess
from typing import List, Literal, Mapping

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class CodeSearchInput(BaseModel):
    type: Literal["CodeSearchInput"] = "CodeSearchInput"
    directory: str = Field(
        ...,
        description="A single absolute path to a directory you want to explore.",
    )
    # regexes: List[str] = Field(
    #     ...,
    #     description="A list of regex patterns that will be applied to files within the corresponding directories. Must match number of directories since each regex corresponds to a directory. Default to '.' to match everything.",
    # )


class CodeSearchTool(BaseTool):
    name: str = "ripgrep_multi_search"
    description: str = (
        "Searches multiple directories using ripgrep (rg). Accepts lists for directories, regex patterns, and allowed file extensions. "
        "Returns a dictionary mapping file paths to their full source code contents."
    )
    args_schema: type[BaseModel] = CodeSearchInput

    def _run(
        self,
        directories: List[str],
        allowed_extensions: List[str] = [
            ".py",
            ".ts",
            ".js",
            ".java",
            ".go",
            ".rs",
            ".yaml",
        ],
    ) -> str:
        # Ensure that the two lists have the same length.
        # if len(directories) != len(regexes):
        #     raise ValueError("The number of directories and regexes must match.")
        regexes = ["."] * len(directories)

        result_map = {}

        # Pair directories and regexes by their index
        for directory, regex in zip(directories, regexes):
            print(f"Searching in {directory} with regex: {regex}")
            # Construct ripgrep command
            command = ["rg", regex, directory]
            proc = subprocess.run(command, capture_output=True, text=True)

            # rg returns exit code 1 when no matches are found.
            if proc.returncode not in [0, 1]:
                print(f"Error running rg in {directory}: {proc.stderr}")
                continue

            # Process the output: expected format "filepath:content"
            file_contents = {}
            for line in proc.stdout.splitlines():
                if ":" in line:
                    file_path, _, content = line.partition(":")
                    if file_path not in file_contents:
                        file_contents[file_path] = []
                    file_contents[file_path].append(content)

            # For each file, check allowed extension and read the full content (avoiding duplicate processing).
            for file_path in file_contents:
                # If allowed_extensions is specified, filter out files with extensions not in the list.
                if allowed_extensions:
                    ext = os.path.splitext(file_path)[1]
                    if ext not in allowed_extensions:
                        continue

                if file_path in result_map:
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        result_map[file_path] = f.read()
                except Exception as e:
                    result_map[file_path] = f"Error reading file: {e}"

        return self.pretty_format_code_map(result_map)

    async def _arun(
        self,
        directories: List[str],
        regexes: List[str],
        allowed_extensions: List[str] = [],
    ) -> Mapping[str, str]:
        raise NotImplementedError(
            "Asynchronous execution is not supported for CodeSearchTool."
        )

    def pretty_format_code_map(self, code_map: Mapping[str, str]) -> str:
        """
        Formats the mapping of file paths to code contents into a neatly formatted string.
        """
        formatted_output = []
        for file_path, content in code_map.items():
            header = f"File: {file_path}"
            separator = "-" * len(header)
            formatted_output.append(f"{header}\n{separator}\n{content}\n")
        return "\n".join(formatted_output)


if __name__ == "__main__":
    # Example input: two lists of directories, regex patterns, and allowed file extensions.
    directories = [
        "./oncall/code",  # Directory to search for code files.
        "./oncall/logs",  # Directory to search for log files.
    ]
    regexes = [
        ".",  # Regex to match everything in the first directory.
        ".",  # Regex to match everything in the second directory.
    ]
    # Only include files ending with '.py' or '.txt'
    allowed_extensions = [".py", ".txt"]

    tool = CodeSearchTool()
    results = tool.run(
        {
            "directories": directories,
            "regexes": regexes,
            "allowed_extensions": allowed_extensions,
        }
    )

    print("\n=== Ripgrep Search Results ===")
    print(json.dumps(results, indent=2))

    print("\n=== Pretty Formatted Output ===")
    print(tool.pretty_format_code_map(results))
