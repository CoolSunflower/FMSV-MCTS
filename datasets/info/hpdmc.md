# High Performance Dynamic Memory Controller

## Signal Mapper Output (LLM2)

```
[sys_clk]: System clock signal. Drives the core logic of the memory controller, used for synchronizing operations.
[sys_clk_n]: Complementary (inverted) system clock signal. Used alongside `sys_clk` to provide differential clocking.
[dqs_clk]: Data Strobe Clock. Used in DDR interfaces to align data transfers with memory clock edges.
[dqs_clk_n]: Complementary Data Strobe Clock. Provides inverted clocking for differential data transfer.
[sys_rst]: System Reset. Resets internal logic to ensure proper start-up and initialization of the memory controller.
[csr_a]: Address for CSR (Control and Status Register). Used for identifying specific registers within the memory controller during configuration.
[csr_we]: Write Enable for CSR. Indicates when a write operation to CSR registers is active.
[csr_di]: Data Input for CSR. Represents data coming from the CPU to configure or control specific settings within the memory controller.
[csr_do]: Data Output for CSR. Used by the CPU to read the current status or configuration from the memory controller.
[fml_adr]: Address input for Fast Memory Link (FML). Points to specific memory addresses in the SDRAM.
[fml_stb]: Strobe signal for FML transactions. Signals a valid transaction on the FML bus.
[fml_we]: Write Enable for FML. Specifies whether the current FML transaction is a read (inactive) or write (active) operation.
[fml_ack]: Acknowledge signal for FML transactions. Indicates successful completion of a memory operation.
[fml_di]: Data Input for FML. Carries data from CPU to SDRAM through FML for write operations.
[fml_do]: Data Output for FML. Holds data read from SDRAM that is sent to the CPU.
[sdram_cke]: SDRAM Clock Enable. Controls the SDRAM clocking for power-saving modes and initialization sequences.
[sdram_cs_n]: SDRAM Chip Select (active low). Enables or disables the SDRAM chip during operations.
[sdram_we_n]: SDRAM Write Enable (active low). Controls write operations to SDRAM, in combination with other signals.
[sdram_cas_n]: SDRAM Column Address Strobe (active low). Selects specific columns in SDRAM for data access.
[sdram_ras_n]: SDRAM Row Address Strobe (active low). Selects specific rows in SDRAM for data access.
[sdram_adr]: SDRAM Address bus. Carries the row and column addresses during SDRAM operations.
[sdram_ba]: SDRAM Bank Address. Selects the specific bank within SDRAM memory.
[sdram_dm]: SDRAM Data Mask. Used to mask or ignore certain bytes during a write operation to SDRAM.
[dqs_psen]: DQS Phase Shift Enable. Allows the CPU to adjust the phase of the DQS signal for timing alignment in data transfers.
[dqs_psincdec]: DQS Phase Shift Increment/Decrement. Allows fine adjustments to the phase shift of DQS.
[dqs_psdone]: DQS Phase Shift Done. Indicates when the DQS phase shift adjustment is complete.
[pll_stat]: PLL Status. Monitors the Phase-Locked Loop (PLL) status, ensuring stable clock signals for SDRAM operations.
```

## SPEC Analyser (LLM1)
The SPEC Analyser output for each clock signal is given in hpdmc.csv

## Waveform Analyser (LLM3)
This specification does not have any waveforms

## Adding Workflow/Architecture Information using SPEC Analyser
The architecture diagram was provided to the LLM.ss