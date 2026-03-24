# memground_agent/env/env_factory.py
"""Game environment factory"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from memground_agent.common.config import EnvConfig
from memground_agent.env.base_env import BaseGameEnv
from memground_agent.env.base_prompt_builder import BasePromptBuilder
from memground_agent.env.base_game_utils import BaseGameUtils
from memground_agent.env.type_help_env import TypeHelpEnv, TypeHelpConfig
from memground_agent.env.no_case_should_remain_unsolved_env import NoCaseEnv, NoCaseConfig
from memground_agent.env.trpg_env import TRPGEnv, TRPGConfig


def create_game_env(env_config: EnvConfig, root_path: Path) -> BaseGameEnv:
    """Create a game environment based on configuration

    Args:
        env_config: Environment configuration
        root_path: Project root directory

    Returns:
        Game environment instance
    """
    game_type = env_config.game_type.lower()

    if game_type == "type_help":
        # Type Help puzzle game
        # Extract data_path from scenes_path (strip nodes.json)
        scenes_path = Path(env_config.scenes_path)
        data_path = root_path / scenes_path.parent
        config = TypeHelpConfig(
            game_type="type_help",
            data_path=data_path,
            start_node_id=env_config.start_node_id,
            test_language=env_config.test_language,
            enable_hint=env_config.enable_hint,
            hint_failure_threshold=env_config.hint_failure_threshold
        )
        return TypeHelpEnv(config)

    elif game_type == "no_case_should_remain_unsolved":
        # No Case Should Remain Unsolved
        scenes_path = Path(env_config.scenes_path)
        data_path = root_path / scenes_path.parent
        config = NoCaseConfig(
            game_type="no_case_should_remain_unsolved",
            data_path=data_path,
            start_node_id=env_config.start_node_id,
            test_language=env_config.test_language
        )
        return NoCaseEnv(config)

    elif game_type == "trpg":
        # TRPG game
        config = TRPGConfig(
            game_type="trpg",
            data_path=root_path / env_config.data_dir,
            qa_path=root_path / env_config.qa_dir,
            story_name=env_config.story_name,
            test_language=env_config.test_language,
        )
        return TRPGEnv(config)

    else:
        raise ValueError(f"Unknown game type: {game_type}")


def create_prompt_builder(env_config: EnvConfig, goal_instruction: str) -> BasePromptBuilder:
    """Create a Prompt builder for the given game type

    Args:
        env_config: Environment configuration
        goal_instruction: Game goal instruction

    Returns:
        Game-specific Prompt builder instance
    """
    game_type = env_config.game_type.lower()

    if game_type == "type_help":
        # Import the Type Help game prompt builder
        from env.type_help.prompt_builder import TypeHelpPromptBuilder
        return TypeHelpPromptBuilder(
            goal_instruction,
            test_language=env_config.test_language,
            provide_naming_rules=getattr(env_config, 'provide_naming_rules', False),
        )

    elif game_type == "no_case_should_remain_unsolved":
        # Import the No Case Should Remain Unsolved game prompt builder
        from env.no_case_should_remain_unsolved.prompt_builder import NoCasePromptBuilder
        return NoCasePromptBuilder(
            goal_instruction,
            test_language=env_config.test_language,
            show_order_judgements_history=env_config.show_order_judgements_history
        )

    elif game_type == "trpg":
        # Import the TRPG game prompt builder
        from env.trpg.prompt_builder import TRPGPromptBuilder
        return TRPGPromptBuilder(goal_instruction)

    else:
        raise ValueError(f"Unknown game type: {game_type}")


def get_supported_game_types() -> list[str]:
    """Get the list of supported game types"""
    return ["type_help", "no_case_should_remain_unsolved", "trpg"]


def create_game_utils(env_config: EnvConfig) -> BaseGameUtils:
    """Create game utilities for the given game type

    Args:
        env_config: Environment configuration

    Returns:
        Game-specific utility instance
    """
    game_type = env_config.game_type.lower()

    if game_type == "type_help":
        # Import the Type Help game utilities
        from env.type_help.utils.game_utils import TypeHelpGameUtils
        return TypeHelpGameUtils()

    elif game_type == "no_case_should_remain_unsolved":
        # Import the No Case Should Remain Unsolved game utilities
        from env.no_case_should_remain_unsolved.utils.game_utils import NoCaseGameUtils
        return NoCaseGameUtils()

    elif game_type == "trpg":
        # Import the TRPG game utilities
        from env.trpg.utils.game_utils import TRPGGameUtils
        return TRPGGameUtils()

    else:
        raise ValueError(f"Unknown game type: {game_type}")
