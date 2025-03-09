# Root Cause Analysis Bot

An intelligent Discord bot designed to assist with customer support by automatically analyzing reported issues, searching relevant code and logs, and determining root causes for problems.

## Overview

This system processes customer support tickets submitted via Discord, leverages AI agents to search through codebases and log data, and provides detailed root cause analysis reports to users. The bot is particularly suited for debugging microservice architectures.

## Key Features

- **Discord Integration**: Receives support tickets directly through Discord messages
- **Intelligent Ticket Processing**: Extracts relevant information from user messages to create structured support tickets
- **Automated Root Cause Analysis**: Uses a state-based agent system to:
  - Search relevant code sections
  - Query logs from appropriate services
  - Analyze gathered information
  - Determine the root cause of reported issues
- **Adaptive Information Gathering**: Can recursively request more code or logs as needed to form a complete understanding
- **Multi-Model Approach**: Uses different LLM models optimized for specific tasks:
  - Fast model for parsing and search operations
  - Reasoning model for in-depth analysis

## Architecture

The system consists of several interconnected components:

1. **Discord Bot Interface** (`bot.py`): Entry point for user interactions
2. **Response Generator** (`chat/response_generator.py`): Processes conversations to extract ticket information
3. **Agent System** (`agent/agentv3.py`): Orchestrates the RCA workflow using a state machine
4. **Search Components**:
   - Code Search: Finds and retrieves relevant code sections
   - Log Search: Queries service logs using Grafana Loki
5. **Analysis Components**:
   - Reasoner: Analyzes code and logs to determine root causes
   - Reviewer: Validates analyses for completeness and accuracy

## Workflow

1. User reports an issue in Discord
2. Bot processes message history to create a structured ticket
3. Agent initiates search for relevant code and logs
4. Reasoner analyzes gathered information
5. If more context is needed, additional searches are performed
6. Once sufficient information exists, a root cause analysis is generated and reviewed
7. Final analysis is delivered back to the user

## Codebase Structure

- **bot.py**: Main entry point and Discord integration
- **agent/**: Core agent system for orchestrating the RCA workflow
- **chat/**: User interaction and support ticket processing
- **code/**: Codebase search and analysis capabilities
- **logs/**: Log query and analysis functionality
- **lib/**: Utility functions and common operations

## Requirements

- Python 3.9+
- Discord API credentials
- Access to microservice codebase
- Grafana Loki for log storage and querying
- OpenAI API access for LLM capabilities
