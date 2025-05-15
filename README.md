# Official Code Implementation for "SANGAM: SystemVerilog Assertion Generation via Monte Carlo Tree Self-Refine".

## Abstract
Recent advancements in the field of reasoning using Large Language Models (LLMs) have created new possibilities for more complex and automatic Hardware Assertion Generation techniques. This paper introduces SANGAM, a SystemVerilog Assertion Generation framework using LLM-guided Monte Carlo Tree Search for the automatic generation of SVAs from industry-level specifications. The proposed framework utilizes a three-stage approach: Stage 1 consists of Multimodal Specification Processing using Signal Mapper, SPEC Analyzer, and Waveform Analyzer LLM Agents. Stage 2 consists of using the Monte Carlo Tree Self-Refine (MCTSr) algorithm for automatic reasoning about SVAs for each signal, and finally, Stage 3 combines the MCTSr-generated reasoning traces to generate SVA assertions for each signal. The results demonstrated that our framework, SANGAM, can generate robust set of SVAs, performing better in the evaluation process in comparison to the recent methods.

## Code structure
This repository contains the code implementation along with the runtime results for rv timer and i2c designs.
The main code file to run the entire assertion generation script is Main({design_name}).ipynb, only the design name needs to be changed at the top. Also ensure that the design specification and rtl is present in the dataset folder, and that jaspergold testing is correctly setup for which you will need to add the setup scripts to run_jg function in utils.py.
