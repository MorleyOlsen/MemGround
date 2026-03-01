# galagent/env/env_factory.py
"""游戏环境工厂"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from galagent.common.config import EnvConfig
from galagent.env.base_env import BaseGameEnv
from galagent.env.base_prompt_builder import BasePromptBuilder
from galagent.env.base_game_utils import BaseGameUtils
from galagent.env.kb_env import KBEnv, KBEnvConfig
from galagent.env.type_help_env import TypeHelpEnv, TypeHelpConfig
from galagent.env.dust_env import DustEnv, DustConfig


def create_game_env(env_config: EnvConfig, root_path: Path) -> BaseGameEnv:
    """根据配置创建游戏环境

    Args:
        env_config: 环境配置
        root_path: 项目根目录

    Returns:
        游戏环境实例
    """
    game_type = env_config.game_type.lower()

    if game_type == "kb":
        # 知识库类型的galgame
        scenes_path = root_path / env_config.scenes_path
        config = KBEnvConfig(
            scenes_path=scenes_path,
            start_node_id=env_config.start_node_id
        )
        return KBEnv(config)

    elif game_type == "type_help":
        # Type Help 解谜游戏
        # 从scenes_path中提取data_path（去掉nodes.json）
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

    elif game_type == "dust":
        # Dust 推理游戏
        scenes_path = Path(env_config.scenes_path)
        data_path = root_path / scenes_path.parent
        config = DustConfig(
            game_type="dust",
            data_path=data_path,
            start_node_id=env_config.start_node_id,
            test_language=env_config.test_language
        )
        return DustEnv(config)

    else:
        raise ValueError(f"Unknown game type: {game_type}")


def create_prompt_builder(env_config: EnvConfig, goal_instruction: str, agent_config=None) -> BasePromptBuilder:
    """根据游戏类型创建对应的Prompt构建器

    Args:
        env_config: 环境配置
        goal_instruction: 游戏目标指令
        agent_config: Agent配置（可选，用于传递功能开关）

    Returns:
        游戏特定的Prompt构建器实例
    """
    game_type = env_config.game_type.lower()

    if game_type == "kb":
        # 导入KB游戏的prompt builder
        from env.kb.prompt_builder import KBPromptBuilder
        return KBPromptBuilder(goal_instruction)

    elif game_type == "type_help":
        # 导入Type Help游戏的prompt builder
        from env.type_help.prompt_builder import TypeHelpPromptBuilder
        return TypeHelpPromptBuilder(
            goal_instruction,
            test_language=env_config.test_language,
            provide_naming_rules=getattr(env_config, 'provide_naming_rules', False),
        )

    elif game_type == "dust":
        # 导入Dust游戏的prompt builder
        from env.dust.prompt_builder import DustPromptBuilder
        return DustPromptBuilder(
            goal_instruction,
            test_language=env_config.test_language,
            show_order_judgements_history=getattr(agent_config, 'show_order_judgements_history', True)
        )

    else:
        raise ValueError(f"Unknown game type: {game_type}")


def get_supported_game_types() -> list[str]:
    """获取支持的游戏类型列表"""
    return ["kb", "type_help", "dust"]


def create_game_utils(env_config: EnvConfig) -> BaseGameUtils:
    """根据游戏类型创建对应的游戏工具

    Args:
        env_config: 环境配置

    Returns:
        游戏特定的工具实例
    """
    game_type = env_config.game_type.lower()

    if game_type == "kb":
        # 导入KB游戏的工具类
        from env.kb.utils.game_utils import KBGameUtils
        return KBGameUtils()

    elif game_type == "type_help":
        # 导入Type Help游戏的工具类
        from env.type_help.utils.game_utils import TypeHelpGameUtils
        return TypeHelpGameUtils()

    elif game_type == "dust":
        # 导入Dust游戏的工具类
        from env.dust.utils.game_utils import DustGameUtils
        return DustGameUtils()

    else:
        raise ValueError(f"Unknown game type: {game_type}")
