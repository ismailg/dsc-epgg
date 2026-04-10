# Hetzner Job Scripts

These scripts are intended to be launched inside a tmux-managed Hetzner run dir.

- `phase3_vecstraight_clean_msgsource_queue_20260410.sh` runs the clean exogenous `msg_source_mode`
  family sequentially (`learned`, `public_random`, `uniform`, `fixed0`, `fixed1`) using the staged
  project checkout on the Hetzner host.
