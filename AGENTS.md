# Repository Rules

## Server access

- Any operation on the remote server or `/data02/qbit/quantum_sim` must use
  the user's fixed guarding allow-list commands.
- Do not use SSH, SCP, SFTP, Orca, desktop control, direct remote shells, or
  another bypass of guarding.
- User messages such as "continue" or "do it" do not expand server access.
- If guarding lacks a required command, stop and request a new fixed command;
  do not look for a side channel.
- Starting, stopping, or cleaning processes must obey guarding approval rules.

## Image and experiment handling

- Do not open or visually inspect generated image files. Plotting scripts may
  create image outputs and validate their existence and file size only.
- Keep experiment execution and plotting separate; plotting must read recorded
  result files and must not rerun an experiment or invent measurements.
