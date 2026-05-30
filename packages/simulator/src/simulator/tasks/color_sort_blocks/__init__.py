import gymnasium as gym


gym.register(
    id="HCIS-ColorSortBlocks-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.color_sort_blocks_env_cfg:ColorSortBlocksEnvCfg",
    },
)
