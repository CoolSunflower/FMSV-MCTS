MODEL_NAME = "deepseek-ai/DeepSeek-R1"
# MODEL_NAME = 'deepseek/deepseek-r1:free'

SVA_SYSTEM_PROMPT = '''
Please act as profession VLSI verification engineer. You will be provided with a specification and workflow information.
Along with that you will be provided with a signal name along with its specification.
Please write all the cooresponding SVAs based on the defined Verilog signals that benefit both the RTL design and verification processes.
Do not generate any new signals. Do NOT make any new information that is not provided. Make sure that the generate SVAs have no syntax error, and strictly follow the function of the given specification/description.
The generated SVAs should include but definately not be limited to the following types:
[width]: Check signal width using the $bits operator.
[connectivity]: Check if the signal can be correctly exercised and also the value propogation among all connected signals.
[function]: Check if the function defined in the specification is implemented as expected.
Make sure the SVAs generated are CORRECT, CONSISTENT, and COMPLETE.
You are allowed to add more types of SVAs than above three if needed, but ensure you do not create a new signal than the specification.
Sometimes you may also be provided with improvement feedback, take it into consideration and improve your SVAs while maintaing CORRECTNESS, CONSISTENCY and COMPLETENESS.
Lets think step by step. Keep your answers short and to the point. If some information is not avalaible you can skip it. Don't overthink! Do not go into thinking loops and waste your time, if you have enough evidence to validate a hypothesis then run with that hypothesis.
Generate assertions ONLY and ONLY for the current signal! Dont overthink. Whatever you output in the thinking process will not be considered the final output, ensure you give your final output outside the think tags.
Note that the else construct is NOT supported.
Format of each SVA assertion:
Either single line assertion:
example: assert property (@(posedge <CLOCK NAME>) <PROPERTY>);  
Or Multi line assertion example:
property <PROPERTY_NAME>;  
  @(posedge <CLOCK NAME>)  
  <PROPERTY>
endproperty  
assert property (<PROPERTY_NAME>);  
Always generate the final assertions between the ```systemverilog and ``` tags. Parameters are already added in code so you dont need to generate them. The only parameters you can add are localparam. Please atleast include all the three types of assertions mentioned above: width, connectivity and functionality. Put the width assertion even if it is 1 bit.
'''

CRITIC_SYSTEM_PROMPT = '''
Please act as a critic to a professional VLSI verification engineer. You will be provided with a specification and workflow information.
Along with that you will be provided with a signal name, its specification and the SVAs generate by a professional VLSI verification engineer for that signal.
Analyse the SVAs strictly and criticise everything possible. You are NOT allowed to create new signal names apart from the specification.
You are NOT allowed to come up with new information other than what is provided.
Point out every possible flaw and improvement and give a score from -100 to 100. Also focus on Clock Cycle Misinterpretations, and nested if-else and long conditions.
Be very strict, ensure CORRECTNESS, CONSISTENCY and COMPLETENESS of the generated SVAs. Focus a lot on these 3 things while grading the SVAs and providing feedback.
Lets think step by step. Keep your answers short and to the point. Ensure that the feedback is ONLY and ONLY for the current signal. Do not generate the SVAs only the feedback!
You also need to ensure that the generated assertions are actually consistent with the specification. Ensure that you reward creative thinking slightly.
'''

COMBINER_PROMPT = '''
Please act as a professional VLSI verification engineer. You will be provided with a specification and workflow information, along with that you will be provided with reasoning regarding the generation of SVAs for a particular signal.
Your task is to generate all the SVAs for the signal using all the information provided as data as well as any other reasoning that you can do.
Your major task is to ensure the CORRECTNESS, CONSISTENCY and COMPLETENESS of the generated SVAs.
Here is your answer format: [REASONING]... [SVAs]...
Ensure the best possible reasoning along with generation. Ensure that you capture as many assertions as possible.
'''

I2C_sv_template = '''
module i2c_master_sva (
    input  logic         wb_clk_i,    // Master clock input
    input  logic         wb_rst_i,    // Synchronous reset (active high)
    input  logic         arst_i,      // Asynchronous reset input
    input  logic         wb_stb_i,    // Strobe / Core select input
    input  logic         wb_ack_o,    // Bus cycle acknowledge output
    input  logic         wb_inta_o,   // Interrupt signal output
    input  logic [2:0]   wb_adr_i,    // Lower address bits for Wishbone interface
    input  logic [7:0]   wb_dat_i,    // Data input to the core
    input  logic         wb_cyc_i,    // Valid bus cycle input
    input  logic [7:0]   wb_dat_o,    // Data output from the core
    input  logic         wb_we_i,     // Write enable signal
    input  logic         scl_pad_i,   // Serial Clock (SCL) line input
    input  logic         scl_pad_o,   // Serial Clock (SCL) line output
    input  logic         sda_pad_i,   // Serial Data (SDA) line input
    input  logic         sda_pad_o,   // Serial Data (SDA) line output
    input  logic         scl_pad_oe,  // Serial Clock (SCL) output enable
    input  logic         sda_pad_oe,  // Serial Data (SDA) output enable
    input  logic [7:0]   ctr,         // Control register
    input  logic [7:0]   sr,          // Status register
    input  logic [15:0]  prer,        // Clock Prescale Register
    input  logic [7:0]   txr,         // Transmit register
    input  logic [7:0]   rxr,         // Receive register
    input  logic [7:0]   cr           // Command register
);

parameter ARST_LVL = 1'b0;

{properties}

endmodule
'''

HPDMC_sv_template = '''
module hpdmc_sva (
    input sys_clk,
    input sys_clk_n,
    input sys_rst,
    input dqs_clk,
    input dqs_clk_n,

    /* FML Interface */
    input [25:0] fml_adr,
    input fml_stb,
    input fml_we,
    input fml_ack,
    input [7:0] fml_sel,
    input [63:0] fml_di,
    input [63:0] fml_do,

    /* SDRAM Interface */
    input sdram_cke,
    input sdram_cs_n,
    input sdram_we_n,
    input sdram_cas_n,
    input sdram_ras_n,
    input [12:0] sdram_adr,
    input [1:0] sdram_ba,

    /* DQS & Delay Control */
    input dqs_psen,
    input dqs_psincdec,
    input dqs_psdone,

    /* PLL Status */
    input [1:0] pll_stat,

    /* Control Interface */
    input [31:0] csr_do,
    input [13:0] csr_a,
    input csr_we,
    input [31:0] csr_di,

    /* Data Path Control */
    input direction,
    input direction_r,
    input data_ack
);

{properties}

endmodule
'''

RV_sv_template = '''
module rv_timer_assertions #(
    parameter int N = 1
) (
    input logic clk_i,
    input logic rst_ni,
    input logic active,
    input logic [11:0] prescaler,
    input logic [7:0] step,
    input logic tick,
    input logic [63:0] mtime_d,
    input logic [63:0] mtime,
    input logic [63:0] mtimecmp [N],
    input logic [N-1:0] intr
);

{properties}

endmodule
'''