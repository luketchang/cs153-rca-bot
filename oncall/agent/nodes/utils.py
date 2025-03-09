from oncall.code.common import collect_files, generate_tree_string
from oncall.constants import GRAFANA_URL
from oncall.logs.labels import build_labels_map


def load_file_tree(repo_path) -> str:
    # Define allowed file extensions for code files
    allowed_extensions = (
        ".py",
        ".js",
        ".ts",
        ".java",
        ".go",
        ".rs",
        ".yaml",
        ".cs",
    )
    repo_file_tree, _ = collect_files(
        repo_path, allowed_extensions, repo_root=repo_path
    )
    file_tree_str = generate_tree_string(repo_file_tree)

    return file_tree_str


def format_labels_map(labels_map) -> str:
    result = "Available Loki log labels and values:\n"
    for label, values in labels_map.items():
        result += f"  {label}: {', '.join(values[:10])}"
        if len(values) > 10:
            result += f" (and {len(values) - 10} more)"
        result += "\n"
    return result


def get_formatted_labels_map(start, end, base_url=GRAFANA_URL) -> str:
    labels_map = build_labels_map(base_url, start, end)
    return format_labels_map(labels_map)
