#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime, timedelta

from oncall.agent import run_agent
from oncall.code.common import generate_tree_string, collect_files
from oncall.lib.logging import logger


def main():
    parser = argparse.ArgumentParser(
        description="Run an on-call agent to help debug production issues"
    )

    parser.add_argument(
        "--issue",
        required=True,
        help="Description of the production issue",
    )
    
    parser.add_argument(
        "--codebase_path", 
        help="Path to the codebase to analyze. If not provided, the current directory is used.",
        default=os.getcwd()
    )
    
    parser.add_argument(
        "--walkthrough_path",
        help="Path to a text file containing a walkthrough of the codebase",
    )
    
    parser.add_argument(
        "--log_query",
        help="Initial Loki log query to run (e.g. '{job=~\"default/auth\"}')",
    )
    
    parser.add_argument(
        "--start_time",
        help="Start time for log search in format 'YYYY-MM-DD HH:MM:SS'",
        default=(datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
    )
    
    parser.add_argument(
        "--end_time",
        help="End time for log search in format 'YYYY-MM-DD HH:MM:SS'",
        default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    args = parser.parse_args()

    try:
        # Load walkthrough
        walkthrough = ""
        if args.walkthrough_path:
            with open(args.walkthrough_path, "r", encoding="utf-8") as f:
                walkthrough = f.read()
                logger.info("Using walkthrough from file")
        else:
            # Use a basic walkthrough if none provided
            walkthrough = "This is a microservices-based system with modules for authentication, payments, and more."
            logger.info("Using default walkthrough")
            
        # Generate file tree
        allowed_extensions = (".py", ".js", ".ts", ".java", ".go", ".rs", ".yaml")
        repo_file_tree, _ = collect_files(
            args.codebase_path, allowed_extensions, repo_root=args.codebase_path
        )
        file_tree_str = generate_tree_string(repo_file_tree)
        logger.info("File tree generated successfully")
        
        # Run the agent
        logger.info("Running agent...")
        result = run_agent(
            issue_description=args.issue,
            walkthrough=walkthrough,
            file_tree=file_tree_str,
            log_query=args.log_query
        )
        
        # Print the final response
        print("\n=== Agent Response ===\n")
        for message in result:
            if hasattr(message, "content") and message.type == "ai":
                print(message.content)
                print("\n" + "-" * 80 + "\n")
                
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()